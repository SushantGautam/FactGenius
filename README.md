# KG-LLM: Zero-Shot Prompting and Fuzzy Relation Mining for Superior Reasoning on Knowledge Graphs
## Sushant Gautam's submission for IN5550 final exam: Fact-checking track


# Training fact-checker model
```python
usage: python train_hf.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--model MODEL] [--epochs EPOCHS] [--freeze FREEZE] [--dbpedia_path DBPEDIA_PATH]
                   [--data_path DATA_PATH] [--plot_roc]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --lr LR
  --model MODEL
  --epochs EPOCHS
  --freeze FREEZE
  --dbpedia_path DBPEDIA_PATH
  --data_path DATA_PATH
  --plot_roc            If set, the ROC curve will be plotted and saved.
```
