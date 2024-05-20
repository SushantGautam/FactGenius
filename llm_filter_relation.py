import pickle
import os
import ast
from kg import KG
import numpy as np
import pandas as pd
import glob
from argparse import ArgumentParser

from multiprocessing import Pool
from functools import partial



from openai import OpenAI
from retrying import retry
import re
import json
import ast

client = OpenAI(
    api_key= "EMPTY",
    base_url="http://g002:8000/v1",
)

@retry(stop_max_attempt_number=10, wait_fixed=0)
def call_llm(claim, entities):
    entities_ = {key.replace('"', ''): value for key, value in entities.items()}
    entity_string = '\n\n'.join([f'''Entity-{index}: "{key}" --> {", ".join(value)}''' for index, (key, value) in enumerate(entities_.items(), start=1)])
    output_expectations= "{\n\n" + "".join([f'''"{entity}": ["..." , "...", ... ],  # options (strictly choose from): ''' + " , ".join(connections) + "\n\n" for entity, connections in entities_.items()]) + "}"
    
    content = f'''
    Claim1:
    {claim}
    '''
    # Entity--> Connections:
    # {entity_string} '''

    message= [{"role": "system", "content": 
    '''
    You are an intelligent graph connection finder. You are given a single claim and connection options for the entities present in the claim. Your task is to filter the Connections options that could be relevant to connect given entities to fact-check Claim1. ~ ( tilde ) in the beginning means the reverse connection. '''
    },{"role": "user", "content": content+ '''

    ## TASK:
    - For each of the given entities given in the DICT structure below: 
       Filter the connections strictly from the given options that would be relevant to connect given entities to fact-check Claim1.
    - Think clever, there could be multi-step hidden connections, if not direct, that could connect the entities somehow.
    - Prioritize connections among entities and arrange them based on their relevance. Be extra careful with ~ signs.
    - No code output. No explanation. Output only valid python DICT of structure:\n'''+ output_expectations},]
    # print(message[1]['content'])

    chat_response = client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=message)

    text= chat_response.choices[0].message.content
    # print("\n\n-------------\n\n", claim, entities, ": \n")
    # print(index, text)
    # breakpoint()
    if ("{}" in text) or (len(text)<2): raise IOError("Redo, empty json is in the keys")
    data = ast.literal_eval(re.findall(r'\{.*?\}', text, re.DOTALL)[0])
    if any([f"entity-" in key.lower() for key in data.keys()]): raise IOError("Redo, Entity- is in the keys")
    return data

parser = ArgumentParser()
parser.add_argument("--data_path", default="/global/D1/projects/HOST/Datasets/factKG_ifi/full/")
parser.add_argument("--dbpedia_path",default="/global/D1/projects/HOST/Datasets/factKG_ifi/dbpedia/dbpedia_2015_undirected_light.pickle")
parser.add_argument("--set", choices=["test", "train", "val"], default="train")
parser.add_argument("--num_proc", type=int, default=10)

args = parser.parse_args()
print(args)

kg = KG(pickle.load(open(args.dbpedia_path, 'rb')))
df = pd.read_csv(args.data_path + f'{args.set}.csv')

output_dir = f"llm_{args.set}"
os.makedirs(output_dir, exist_ok=True)
dfx = df[~df.index.isin([int(f.split('/')[-1].split('.')[0]) for f in glob.glob(f'{output_dir}/**.json', recursive=True)])]
# dfx= df.sample(200)
print("Total rows to process", len(dfx))

def process_row(index, row):
    entities = ast.literal_eval(row["Entity_set"])
    save_json_as = f"{output_dir}/{index}.json"
    if os.path.exists(save_json_as):
        return
    resolved_json = call_llm(row['Sentence'], {e: list(kg.kg.get(e,{}).keys()) for e in entities})
    print(index, resolved_json)
    with open(save_json_as, 'w') as f:
        json.dump(resolved_json, f, ensure_ascii=False)


partial_process_row = partial(process_row)
with Pool(processes=args.num_proc) as pool:
    pool.starmap(partial_process_row, dfx.iterrows())

# for index, row in dfx.iterrows():
#     process_row(index, row)

# Model served with vllm 
#python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 2