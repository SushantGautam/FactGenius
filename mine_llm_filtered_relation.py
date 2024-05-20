import glob
import os
import json
import pickle
import os
import ast
from kg import KG
import pandas as pd
import tqdm
from thefuzz import process

from argparse import ArgumentParser


# script to check max number of braches for each node
# max([max([ len(x) for x in list(ast.literal_eval(e).values())]) for e in df.Evidence])

# for file in jsons:
#     data = open(file).read()
#     if "{}" in data.lower() or "entity-" in data.lower() or len(data) < 2:
#         print(data)
#         os.remove(file)
#         print(file)

def fuzzy_matchEntities(A, B, _data):
    results = {}
    for item in B:
        best_match = process.extractOne(str(item), A)
        results[item] = best_match
    resolved  = {key: value for key, value in results.items() if value[1] >= max(v[1] for v in results.values())}
    return {resolved.get(key, (key,0))[0]:value for key, value in _data.items()}


# kg_sorted = {}
# pickle.dump(kg_sorted, open("kg_sorted.pickle", "wb"))
# kg_sorted = pickle.load(open("kg_sorted.pickle", "rb"))
paths_to_str = lambda paths: "|".join([",".join(path) for path in paths])
paths_to_str2 = lambda paths: [evidence[0]+" >- "+evidence[1]  +  " -> "+evidence[2] if "~" not in evidence[1] else evidence[2]+" >- "+evidence[1][1:]+" -> "+evidence[0] for evidence in paths]
# path_string= " | ".join([path for typ in ["connected", "walkable"] for path in paths_to_str2(kg.search(sorted(sorted(A.keys())), probable_evidences)[typ])])



def validateRelation(A,_row=None, _kg=None, skip_second_level=False):
    probable_evidences = {}
    for key, values in A.items():
        all_possible = list(_kg.kg.get(key,{}).keys())
        # all_possible = list(kg_sorted.get(key,[]))
        for value in values:
            tentative_matches = process.extract(str(value), all_possible, limit=2) # [('New York Jets', 100), ('New York Giants', 78)]
            filtered_matches = [match[0] for match in tentative_matches if match[1] > 90]
            probable_evidences[key] = sorted(set(filtered_matches))
    all_connection = sorted(set([word for sentence in probable_evidences.values() for word in sentence]))
    if not skip_second_level:
    ## Second level : to scan for similar connection type
        for key, values in probable_evidences.items():
            all_possible = list(_kg.kg.get(key,{}).keys())
            # all_possible = list(kg_sorted.get(key,[]))
            for value in all_connection:
                tentative_matches = process.extract(str(value), all_possible, limit=2)
                filtered_matches = [match[0] for match in tentative_matches if match[1] > 90]
                probable_evidences[key] = sorted(set(probable_evidences[key]).union(set(filtered_matches)))
    return {k:[[e] for e in v] for k, v in probable_evidences.items()}
    # su =kg.search(ast.literal_eval("['New_York', 'French_language']"), ast.literal_eval("{'New_York': [['language']], 'French_language': [['~language']]}"))
    # paths_to_str(su["connected"])
    # os.system('clear')
    # if _row:
    #     print("Question: ", _row.Sentence, row.Label, "\n real:", sorted(ast.literal_eval(_row["Evidence"]).items()))


    # entities = ast.literal_eval(row["Entity_set"])
    # rels = ast.literal_eval(row["Evidence"])
    # paths_dict = kg.search(entities, rels)
    # connected_paths_str = paths_to_str(paths_dict["connected"])
    # breakpoint()
    # print("Probable:", sorted(probable_evidences.items()))
    # breakpoint()

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data_path", default="/global/D1/projects/HOST/Datasets/factKG_ifi/full/")
    parser.add_argument("--dbpedia_path",default="/global/D1/projects/HOST/Datasets/factKG_ifi/dbpedia/dbpedia_2015_undirected_light.pickle")
    parser.add_argument("--set", choices=["test", "train", "val"], default="train")
    parser.add_argument("--num_proc", type=int, default=10)
    parser.add_argument("--outputPath", default="./llm_v1/")
    parser.add_argument("--skip_second_stage", action="store_true", help="If set, the second stage of relation validation will be skipped.")
    parser.add_argument("--jsons_path", default="./llm_v1_jsons/", help="Path to the jsons files")

    args = parser.parse_args()
    print(args)
    skip_second_stage =  True if args.skip_second_stage else False
    
    
    df = pd.read_csv(args.data_path + f'{args.set}.csv')
    jsons = glob.glob(f'{args.jsons_path}llm_{args.set}/**/*.json', recursive=True)
    print("Total rows to process", len(df))

    kg = KG(pickle.load(open(args.dbpedia_path, 'rb')))
    output_file_name = args.outputPath + f"{args.set}.csv"
    print("Processsed files will be saved in ", output_file_name)
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    sentence_label=[]
    for file in tqdm.tqdm(jsons):
        file_id = int(file.split('/')[-1].split('.')[0])
        try:
            data = json.load(open(file))
            row = df.iloc[file_id]
            true_entities = ast.literal_eval(row["Entity_set"])
            predicted_entities= [k for k in data.keys() if data[k] != []]
            resolved_entities = fuzzy_matchEntities(true_entities, predicted_entities, data)
            resolved_entities_relation= validateRelation(resolved_entities, row, kg, skip_second_level=skip_second_stage)
            kg_results= kg.search(sorted(sorted(resolved_entities_relation.keys())), resolved_entities_relation)
            path_string= " | ".join([path for typ in ["connected", "walkable"] for path in paths_to_str2(kg_results[typ])])
            new_input= f"Claim: {row.Sentence} Evidence: {path_string}"
            sentence_label.append((file_id, new_input, row.Label))
        except Exception as e:
            print(e)
            breakpoint()

    # save sentence_label as Sentence,Label csv file 
    df = pd.DataFrame(sentence_label, columns=["rowID", "Sentence", "Label"]).sort_values(by=["rowID"]).drop(columns=["rowID"])
    df.to_csv(output_file_name, index=False)