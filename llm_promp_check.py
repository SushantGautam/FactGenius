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

from process_llm_filterrelation import fuzzy_matchEntities, validateRelation, paths_to_str2

from openai import OpenAI
from retrying import retry
import re
import json
import ast

@retry(stop_max_attempt_number=10, wait_fixed=0)
def call_llm(row, evidences):
    claim= row.Sentence

    if not args.llm_knowledge:
        instruction_head = '''
    You are an intelligent fact-checker. You are given a single claim and supporting evidence for the entities present in the claim, extracted from a knowledge graph. 
    Your task is to decide whether all the facts in the given claim are supported by  given evidences..
    '''
    else:
        instruction_head = '''
    You are an intelligent fact checker trained on Wikipedia. You are given a single claim and your task is to decide whether all the facts in the given claim are supported by the given evidence using your knowledge.
    '''

    if not args.llm_knowledge:
        content= f'''
    ## TASK:
    Now let’s verify the Claim based on the evidences.
    Claim:  {claim}
    
    Evidences: 
    {evidences}

    '''
    else:
        content= f'''
    ## TASK:
    Now let’s verify the Claim.
    Claim:  {claim}

    '''
    content+= '''
    #Answer Template: 
    "True/False (single word answer),
    One-sentence evidence."
    '''
    
    message= [{"role": "system", "content": 
    instruction_head + '''
    Choose one of {True, False}, and output the one-sentence explanation for the choice. 
    '''
    },{"role": "user", "content": content}]
    # print(message[1]['content'])
    # breakpoint()

    chat_response = client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=message)

    text= chat_response.choices[0].message.content
    first_line = text.split("\n")[0].strip()
    # make sure 'True' or 'False' is in the sub-string of first line
    if not any([i in first_line for i in ["True", "False"]]):
        print("retry")
        raise IOError("True/False not in the first line")
    output_decision = True if "True" in first_line else False
    return text, output_decision

parser = ArgumentParser()
parser.add_argument("--data_path", default="/fp/projects01/ec30/factkg/full/")
parser.add_argument("--dbpedia_path",default="/fp/projects01/ec30/factkg/dbpedia/dbpedia_2015_undirected_light.pickle")
parser.add_argument("--evidence_path", default="/home/sushant/D1/Assignments/in5550_2024/exam/fact-checking/baseline/llm_full_v1", help="Path to the edvidence JSONs predicted by LLM.")
parser.add_argument("--set", choices=["test", "train", "val"], default="train")
parser.add_argument("--num_proc", type=int, default=10)
parser.add_argument("--llm_knowledge", action="store_true", help="If set, the instruction will be claim only LLM based fact checking.")
parser.add_argument("--vllm_url", default="http://g002:8000", help="URL of the vLLM server, e.g., http://g002:8000")

args = parser.parse_args()
print(args)

client = OpenAI(
    api_key= "EMPTY",
    vllm_url= args.vllm_url + "/v1",
)


kg = KG(pickle.load(open(args.dbpedia_path, 'rb')))
df = pd.read_csv(args.data_path + f'{args.set}.csv')

dfx= df
print("Total rows to process", len(dfx))


all_evidence = {}
for file in glob.glob(f'{args.evidence_path}/llm_{args.set}/**.json'):
    idx= int(file.split('/')[-1].split('.')[0])
    if idx in dfx.index:
        with open(file) as f:
            all_evidence[idx]= json.load(f)
    
import multiprocessing
manager = multiprocessing.Manager()
real_predicted = manager.dict()

def process_row(index, row, _shared_dict):
    data = all_evidence[index]

    true_entities = ast.literal_eval(row["Entity_set"])
    predicted_entities= [k for k in data.keys() if data[k] != []]
    resolved_entities = fuzzy_matchEntities(true_entities, predicted_entities, data)
    resolved_entities_relation= validateRelation(resolved_entities, row, kg)
    kg_results= kg.search(sorted(sorted(resolved_entities_relation.keys())), resolved_entities_relation)
    supporting_evidences = "\n".join([path for typ in ["connected", "walkable"] for path in paths_to_str2(kg_results[typ])])
    text, output_decision = call_llm(row, supporting_evidences)
    _shared_dict[index] = [output_decision, text.replace("\n", "|")]
    print(index, output_decision)

partial_process_row = partial(process_row, _shared_dict=real_predicted)
with Pool(processes=args.num_proc) as pool:
    pool.starmap(partial_process_row, dfx.iterrows())

for key, value in real_predicted.items():
    dfx.at[key, 'Predicted'] = value[0]
    dfx.at[key, 'Response'] = value[1]

from sklearn.metrics import classification_report
print(classification_report(list(dfx["Label"].values.tolist()), list(dfx["Predicted"].values.tolist())))

dfx['Metadata'] = [ast.literal_eval(e) for e in dfx.Metatada]

interetsing = ['num1', 'multi claim', 'existence', 'multi hop']
from collections import defaultdict
interetsing_list = defaultdict(list)

for index, row in dfx.iterrows():
    if "negation" in row['Metadata']:
        interetsing_list['negation'].append([row['Label'], row['Predicted']])
        continue
    for each in interetsing:
        if (each in row['Metadata']):
            interetsing_list[each].append([row['Label'], row['Predicted']])

for each in interetsing_list.keys():
    print(f"\nClassification report for {each}")
    print(classification_report([i[0] for i in interetsing_list[each]], [i[1] for i in interetsing_list[each]]))

filtered_df = dfx[['Predicted', 'Response', 'Label']]
if_llm_knowledge = "_llm_knowledge" if args.llm_knowledge else ""
filtered_df.to_csv(f"llm_prompt_check_{args.set}{if_llm_knowledge}.csv", index=True)
print(f"saved to llm_prompt_check_{args.set}{if_llm_knowledge}.csv")

# python llm_promp_check.py --set test --num_proc 50 [--llm_knowledge]