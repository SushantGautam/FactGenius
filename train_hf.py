import os
import random
import pickle
import os
import ast
from kg import KG
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
from transformers import EarlyStoppingCallback, IntervalStrategy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_PROJECT"] = "FactKG_IN9550"
os.environ["WANDB_WATCH"]="false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed_value=2024
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


class FactkgDataset(torch.utils.data.Dataset):
    def __init__(self, df, _kg=None):
        self.inputs = df['Sentence'].tolist()
        if _kg:
            self.paths = self.create_connected_paths(df, _kg)
            self.inputs = [f"{input} {path}" for input, path in zip(self.inputs, self.paths)]
        self.labels = df['Label'].astype(int).tolist()

    def create_connected_paths(self, df, kgx):
        connected_paths = []
        for index, row in df[::-1].iterrows():
            entities = ast.literal_eval(row["Entity_set"])
            rels = ast.literal_eval(row["Evidence"])
            paths_dict = kgx.search(entities, rels)
            connected_paths_str = self.paths_to_str(paths_dict["connected"])
            connected_paths.append(connected_paths_str)
        return connected_paths
    

    def paths_to_str(self, paths):
        path_strings = [",".join(path) for path in paths]
        return "|".join(path_strings)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

def transform_to_dataset(train_df, val_df, test_df, __kg=None):
    train_dataset = FactkgDataset(train_df, __kg)
    val_dataset = FactkgDataset(val_df, __kg)
    test_dataset = FactkgDataset(test_df, __kg)
    train_dataset = Dataset.from_pandas(pd.DataFrame({
        'label_ids': train_dataset.inputs,
        'labels': train_dataset.labels
    }))
    val_dataset = Dataset.from_pandas(pd.DataFrame({
        'label_ids': val_dataset.inputs,
        'labels': val_dataset.labels
    }))
    test_dataset = Dataset.from_pandas(pd.DataFrame({
        'label_ids': test_dataset.inputs,
        'labels': test_dataset.labels
    }))
    return DatasetDict({
    'train': train_dataset, 
    'validation': val_dataset,
    'test': test_dataset
    })

class CollateFunctor:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __call__(self, batch):
        claims_with_evidence, labels = [e['label_ids'] for e in batch], [e['labels'] for e in batch]
        inputs = self.tokenizer(claims_with_evidence, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
        inputs['labels'] = torch.tensor(labels)
        return inputs

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

def compute_metrics(eval_pred):
    logits_, labels = eval_pred
    logits = logits_.argmax(axis=1)
    accuracy = accuracy_score(labels, logits)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, logits, average='binary')
    conf_mat = confusion_matrix(y_true=labels, y_pred=logits)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_mat': conf_mat.tolist(),
    }

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--model", default="roberta-base") # roberta-base:32, bert-base-uncased:64 on V100
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--dbpedia_path",default="/fp/projects01/ec30/factkg/dbpedia/dbpedia_2015_undirected_light.pickle")
parser.add_argument("--data_path", default="/global/D1/projects/HOST/Datasets/factKG_ifi/llm_v1/")
parser.add_argument("--plot_roc", action="store_true", help="If set, the ROC curve will be plotted and saved.")
args = parser.parse_args()
print(args)

kg = KG(pickle.load(open(args.dbpedia_path, 'rb'))) if args.with_evidence else None

tokenizer = AutoTokenizer.from_pretrained(args.model)

# Load data
train_df = pd.read_csv(args.data_path + 'train.csv')
val_df = pd.read_csv(args.data_path + 'val.csv')
test_df = pd.read_csv(args.data_path + 'test.csv')

datasets = transform_to_dataset(train_df, val_df, test_df, kg)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results/"+ args.model.replace("/", "_"),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to='wandb',
    load_best_model_at_end=True,
    # push_to_hub=True,
    # hub_model_id=f"FactKG-{args.model}".replace("/", "_"),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['validation'],
    data_collator=CollateFunctor(tokenizer, 512),
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

# evaluate on test
test_result = trainer.predict(datasets['test'])
print(f"Test results: {test_result.metrics}")



### optional code to print classification report for each interesting category
predictions = test_result.predictions.argmax(axis=1)
from sklearn.metrics import classification_report
print(classification_report(test_result.label_ids, predictions))

interetsing = ['num1', 'multi claim', 'existence', 'multi hop']

dfx = pd.read_csv('/fp/projects01/ec30/factkg/full/test.csv')
dfx['Predicted'] = predictions #index is already same as test.csv

dfx['Label'] = [1 if e == True else 0 for e in dfx.Label]
dfx['Metadata'] = [ast.literal_eval(e) for e in dfx.Metatada]

from collections import defaultdict
interetsing_list = defaultdict(list)

for index, row in dfx.iterrows():
    if "negation" in row['Metadata']:
        interetsing_list['negation'].append([row['Label'], row['Predicted']])
        continue
    for each in interetsing:
        if (each in row['Metadata']):
            interetsing_list[each].append([row['Label'], row['Predicted']])

## for each interesting, calculate the classification_report
for each in interetsing_list.keys():
    print(f"\nClassification report for {each}")
    print(classification_report([i[0] for i in interetsing_list[each]], [i[1] for i in interetsing_list[each]]))

## optional code to plot ROC and Precision-Recall curve
if args.plot_roc:
    fpr, tpr, _ = roc_curve(test_result.label_ids, test_result.predictions[:, 1])
    roc_auc = auc(fpr, tpr)

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(test_result.label_ids, test_result.predictions[:, 1])
    
    # Plot ROC curve and Precision-Recall curve side by side
    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(10, 4))

    # Plot ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')

    plt.tight_layout()
    # save figure
    plt.savefig('roc_curve.png')