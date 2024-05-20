# KG-LLM: Zero-Shot Prompting and Fuzzy Relation Mining for Superior Reasoning on Knowledge Graphs
## Sushant Gautam's submission for IN5550 final exam: Fact-checking track


### Helper function for training fact-checker model
```python
usage: python train_hf.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--model MODEL] [--epochs EPOCHS] [--freeze FREEZE] [--dbpedia_path DBPEDIA_PATH]
                   [--data_path DATA_PATH] [--plot_roc]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --lr LR
  --model MODEL
  --epochs EPOCHS
  --dbpedia_path DBPEDIA_PATH
  --data_path DATA_PATH
  --plot_roc            If set, the ROC curve will be plotted and saved.
```

/global/D1/projects/HOST/Datasets/factKG_ifi/llm_v1/

### Set up Llama3-Instruct inference server using vLLM
####  Run on any server with NVIDIA A100 GPU (80 GB VRAM)
####  Request access to the model at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct. 
####  Make sure you have logged in to Hugging Face with `huggingface-cli whoami` and have access to the model.
```bash
python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-8B-Instruct
```
The server should be up and running on port 8000 at http://hostname:8000/v1 by default.

### 1. Evaluating Zero-shot Claim Only Baseline
```bash
python train_hf.py --data_path
```

### 2. Train and evaluate RoBERTa as Claim Only Baseline
```bash
python train_hf.py 
```