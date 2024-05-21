# KG-LLM: Zero-Shot Prompting and Fuzzy Relation Mining for Superior Reasoning on Knowledge Graphs
## Sushant Gautam's submission for IN5550 final exam: Fact-checking track


### Helper files
#### Helper for fine-tuning and evaluating the fact-checker model
```python
usage: python fine_tune_hf.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--model MODEL] [--epochs EPOCHS] [--freeze FREEZE] [--dbpedia_path DBPEDIA_PATH]
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
The script will output the evaluation results on the test set as well across all five reasoning types as reported on the paper and also save the model in the ./results directory.

####  Helper function for LLM-based fact-checking
```python
usage: python llm_fact_check.py [-h] [--data_path DATA_PATH] [--dbpedia_path DBPEDIA_PATH] [--evidence_path EVIDENCE_PATH] [--set {test,train,val}]
                          [--num_proc NUM_PROC] [--llm_knowledge] [--vllm_url VLLM_URL]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
  --dbpedia_path DBPEDIA_PATH
  --evidence_path EVIDENCE_PATH
                        Path to the edvidence JSONs predicted by LLM.
  --set {test,train,val}
  --num_proc NUM_PROC
  --llm_knowledge       If set, the instruction will be claim only LLM based fact checking.
  --vllm_url VLLM_URL   URL of the vLLM server, e.g., http://g002:8000
```

### Set up Llama3-Instruct inference server using vLLM
#### Note: The output of graph filtering using LLM is made available in the repo already. So you can choose to skip inference server set-up and jump to fine-tuning pre-trained models and evaluating them.
####  Run on any server with NVIDIA A100 GPU (80 GB VRAM)
####  Request access to the model at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct. 
####  Make sure you have logged in to Hugging Face with `huggingface-cli whoami` and have access to the model.

```bash
python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-8B-Instruct
```
The inference server should be up and running on port 8000 at http://hostname:8000 by default. Remember to replace --vllm_url argument with appropriate server url when using LLM. 



### 1. Evaluating Zero-shot Claim Only Baseline with LLM
```bash
python llm_fact_check.py --set test --llm_knowledge --vllm_url http://g002:8000
```

### 2. Fine-tune and evaluate RoBERTa as Claim Only Baseline
```bash
python fine_tune_hf.py --model roberta-base --batch_size 32
```
The fine-tuned model is pushed at: 
https://huggingface.co/SushantGautam/KG-LLM-roberta-base-claim_only.
Can be loaded using the Hugging Face Transformers library and used for inference with appropriate tokenizer.

 The finetuning logs and metrics can be found at https://wandb.ai/ubl/FactKG_IN9550/runs/ui83kr9l.

### 3. Filtering connections and caching data for fine-tuning and evaluation
#### 3.1: Filtering Possible Connections with LLM
```bash
python llm_filter_relation.py --set train --vllm_url http://g002:8000
python llm_filter_relation.py --set val --vllm_url http://g002:8000
python llm_filter_relation.py --set test --vllm_url http://g002:8000
```
This will output JSON files in ./llm_train, ./llm_val, and ./llm_test directories. The output directories have been saved as zip file: llm_v1_jsons.zip.

Assume ./llm_train, ./llm_val, and ./llm_test directories with JSONS are made available inside the ./llm_v1_jsons/ directory. Or can get them by unzipping the llm_v1_jsons.zip file.
```bash
unzip llm_v1_jsons.zip
```

#### 3.2: Fuzzy Relation Mining
##### 3.2.1: Two-stage Fuzzy Relation Mining
```bash
python mine_llm_filtered_relation.py --set train --outputPath ./llm_v1/ --jsons_path ./llm_v1_jsons/
python mine_llm_filtered_relation.py --set val --outputPath ./llm_v1/ --jsons_path ./llm_v1_jsons/
python mine_llm_filtered_relation.py --set test --outputPath ./llm_v1/ --jsons_path ./llm_v1_jsons/
```
This will output CSV files in ./llm_v1/ directory. The output has been made available in the repo already.

##### 3.2.1: Single-stage Fuzzy Relation Mining
```bash
python mine_llm_filtered_relation.py --set train --outputPath ./llm_v1_singleStage/ --jsons_path ./llm_v1_jsons/ --skip_second_stage 
python mine_llm_filtered_relation.py --set val --outputPath ./llm_v1_singleStage/ --jsons_path ./llm_v1_jsons/ --skip_second_stage
python mine_llm_filtered_relation.py --set test --outputPath ./llm_v1_singleStage/ --jsons_path ./llm_v1_jsons/ --skip_second_stage
```
This will output CSV files in ./llm_v1_singleStage/ directory. The output has been made available in the repo already.

### 4. Zero-shot LLM as Fact Classifier with evidence
```bash
python llm_fact_check.py --set test --vllm_url http://g002:8000  --evidence_path ./llm_v1_jsons
```

### 5. Fine-tuning pre-trained models 
#### 5.1: Fine-tuning Two-stage BERT classifier on the filtered data
```bash
python fine_tune_hf.py --model bert-base-uncased --batch_size 64 --data_path ./llm_v1/
```
#### 5.2: Fine-tuning Single-stage RoBERTa classifier on the filtered data
```bash
python fine_tune_hf.py --model roberta-base --batch_size 32 --data_path ./llm_v1_singleStage/
```
#### 5.3: Fine-tuning Two-stage RoBERTa classifier on the filtered data
```bash
python fine_tune_hf.py --model roberta-base --batch_size 32 --data_path ./llm_v1/
```

The fine-tuned models are pushed at: 
https://huggingface.co/SushantGautam/KG-LLM-bert-base, 
https://huggingface.co/SushantGautam/KG-LLM-roberta-base-single_stage and
https://huggingface.co/SushantGautam/KG-LLM-roberta-base.
Can be loaded using the Hugging Face Transformers library and used for inference with appropriate tokenizer.

 The finetuning logs and metrics can be found at https://wandb.ai/ubl/FactKG_IN9550/runs/7sf9kelb, https://wandb.ai/ubl/FactKG_IN9550/runs/sweuj8g6 and https://wandb.ai/ubl/FactKG_IN9550/runs/m5vqnfcr.
