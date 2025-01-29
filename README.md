# Conv2Query

This is the repository for the paper titled **Bridging the Gap: From Ad-hoc to Proactive Search in Conversations**.

This repository is structured into the following parts:
1. [Prerequisites](#1-prerequisites)
   - [1.1 Installation](#11-installation)
   - [1.2 Data preparation](#12-data-preparation)
2. [Producing pseudo ad-hoc query targets for training](#2-producing-pseudo-ad-hoc-query-targets-for-training)
   - [2.1 Generating ad-hoc queries from documents](#21-generating-ad-hoc-queries-from-documents)
   - [2.2 Query filtering based on document relevance and conversation alignment (QF-DC)](#22-query-filtering-based-on-document-relevance-and-conversation-alignment-qf-dc)
     - [2.2.1 Query–document relevance](#221-query-document-relevance)
     - [2.2.2 Query–conversation relevance](#222-query-conversation-relevance)
     - [2.2.3 Optimal query selection](#223-optimal-query-selection)
3. [Learning to generate ad-hoc queries from conversations](#3-learning-to-generate-ad-hoc-queries-from-conversations)
4. [Generating ad-hoc queries for retrieval (inference)](#4-generating-ad-hoc-queries-for-retrieval-inference)
5. [Reusing off-the-shelf ad-hoc retrievers](#5-reusing-off-the-shelf-ad-hoc-retrievers-training)
6. [Further fine-tuning ad-hoc retrievers using filtered ad-hoc queries (optional)](#6-further-fine-tuning-ad-hoc-retrievers-using-filtered-ad-hoc-queries-optional)


## ⚙️ 1. Prerequisites <a name="1-prerequisites"></a>

### 1.1 Installation <a name="11-installation"></a>

Install dependencies:
```bash
pip install -r requirements.txt
```
Please install [Tevatron](https://github.com/texttron/tevatron) in advance.

We directly fetch weights of LLMs from Hugging Face. Please set your own token and your cache directory:
```bash
export TOKEN={your token to use as HTTP bearer authorization for remote files}
export CACHE_DIR={your cache path that stores the weights of LLMs}
```
All experiments are conducted on 4 NVIDIA A100 GPUs (40GB).

### 1.2 Data preparation <a name="12-data-preparation"></a> 

### The [ProCIS](https://dl.acm.org/doi/10.1145/3626772.3657869) dataset (published at SIGIR 2024)
Please download the raw data and then put the raw data in the directory of `./data/procis/raw`:
```bash
mkdir data
mkdir data/procis
mkdir data/procis/raw
mkdir data/procis/corpus  
mkdir data/procis/queries 
mkdir data/procis/qrels
mkdir data/procis/indexes
mkdir data/procis/runs
mkdir data/procis/fillter
mkdir data/procis/training

wget -P ./data/procis/raw https://archive.org/download/procis/procis.zip
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip ./data/procis/raw/procis.zip -d ./data/procis/raw/
```

Next, run the script to preprocess the ProCIS dataset:
```bash
python -u ./preprocess_procis.py
```
The preprocessing will produce TREC-style queries and qrels stored in `data/procis/queries` and `data/procis/qrels`, respectively. And it will produce Pyserini-style and Tevatron-style corpus files stored in `data/procis/corpus`.

### The [WebDisc](https://dl.acm.org/doi/10.1145/3578337.3605139) dataset  (published at ICTIR 2023)
Please ask the original author Kevin Ros (kjros2@illinois.edu) of WebDisc to get the raw data, and then put the data in the directory of `./data/webdisc/raw`. After decompressing, execute the script to preprocess the ProCIS dataset:
```bash
mkdir data/webdisc
mkdir data/webdisc/raw
mkdir data/webdisc/corpus  
mkdir data/webdisc/queries 
mkdir data/webdisc/qrels
mkdir data/webdisc/indexes
mkdir data/webdisc/runs
mkdir data/webdisc/fillter
mkdir data/webdisc/training

tar -xvf ./data/webdisc/raw/webpages_v3.tar.gz -C ./data/webdisc/raw/
```

Next, run the script to preprocess the Webdisc dataset:
```bash
python -u ./preprocess_webdisc.py
```
The preprocessing will produce TREC-style queries and qrels stored in `data/webdisc/queries` and `data/webdisc/qrels`, respectively. And it will produce Pyserini-style and Tevatron-style corpus files stored in `data/webdisc/corpus`.


## 📜 2. Producing pseudo ad-hoc query targets for training <a name="2-producing-pseudo-ad-hoc-query-targets-for-training"></a>

### 2.1 Generating ad-hoc queries from documents <a name="21-generating-ad-hoc-queries-from-documents"></a>

### ProCIS
Please use the following commands to run [Doc2Query-T5](https://huggingface.co/BeIR/query-gen-msmarco-t5-large-v1) to generate 100 ad-hoc queries per relevant document for each conversational context.
Alternatively, we provide the script to run [Doc2Query-Llama2](https://huggingface.co/soyuj/llama2-doc2query) to generate 70 queries per relevant document; we set the number of query to 70 because the GPU memory limitation; our preliminary experiments show that Doc2Query-Llama2 does not offer a noticeable improvement over Doc2Query-T5.
The generated queries will be stored in `data/procis/queries`.
```bash
# Doc2Query-T5
for i in 0 1 2 3
do
gpuid=$((i)) 
CUDA_VISIBLE_DEVICES=${gpuid} \
nohup python -u doct5query.py \
--corpus_dir ./data/procis/corpus/procis.corpus.jsonl/procis.corpus.jsonl \
--qrels_dir ./data/procis/qrels/procis.train-filtered1000.qrels.turn-link.txt \
--output_dir ./data/procis/queries \
--batch_size 2 \
--query_num 100 \
--max_input_length 512 \
--num_chunks 4 \
--local_rank ${i} \
> procis.train.queries.doct5query-100.chunk${i}.log 2>&1 &
done

# Doc2Query-Llama2
for i in 0 1 2 3
do
gpuid=$((i)) 
CUDA_VISIBLE_DEVICES=${gpuid} \
nohup python -u docllamaquery.py \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--corpus_dir ./data/procis/corpus/procis.corpus.jsonl/procis.corpus.jsonl \
--qrels_dir ./data/procis/qrels/procis.train-filtered1000.qrels.turn-link.txt \
--output_dir ./data/procis/queries \
--batch_size 1 \
--query_num 70 \
--max_input_length 512 \
--chunk ${i} \
> procis.train-filtered1000.queries.docllama2query-70-topk10.chunk${i}.log 2>&1 &
done
```

### WebDisc
The following operations are similar to ProCIS. The generated queries will be stored in `data/webdisc/queries`.

```bash
# Doc2Query-T5
for i in 0 1 2 3
do
gpuid=$((i)) 
CUDA_VISIBLE_DEVICES=${gpuid} \
nohup python -u doct5query.py \
--corpus_dir ./data/webdisc/corpus/webdisc.corpus.jsonl/webdisc.corpus.jsonl \
--qrels_dir ./data/webdisc/qrels/webdisc.train.qrels.txt \
--output_dir ./data/webdisc/queries \
--batch_size 2 \
--query_num 100 \
--max_input_length 512 \
--num_chunks 4 \
--local_rank ${i} \
> webdisc.train.queries.doct5query-100.chunk${i}.log 2>&1 &
done

# Doc2Query-Llama2
for i in 0 1 2 3
do
gpuid=$((i)) 
CUDA_VISIBLE_DEVICES=${gpuid} \
nohup python -u docllamaquery.py \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--corpus_dir ./data/webdisc/corpus/webdisc.corpus.jsonl/webdisc.corpus.jsonl \
--qrels_dir ./data/webdisc/qrels/webdisc.train.qrels.txt \
--output_dir ./data/webdisc/queries \
--batch_size 1 \
--query_num 70 \
--max_input_length 512 \
--num_chunks 4 \
--local_rank ${i} \
> webdisc.train.queries.docllama2query-100.chunk${i}.log 2>&1 &
done
```

### 2.2 Query filtering based on document relevance and conversation alignment (QF-DC) <a name="22-query-filtering-based-on-document-relevance-and-conversation-alignment-qf-dc"></a>
For predicting query--document relevance and query--conversation relevance, we use [RepLLaMA](https://huggingface.co/castorini/repllama-v1-7b-lora-passage) as our relevance model. We use the Tevatron package.

#### 2.2.1 Query–document relevance <a name="221-query-document-relevance"></a>

#### ProCIS

Run the following commands to prepare inputs of the relevance model, and conduct query--document relevance prediction on ProCIS.
The relevance score file will be stored in `./data/procis/filter/`.
```bash
mode=q2d
doc_len=512

# generate relevance prediction input file
for i in 0 1 2 3
do
python prepare_rerank_file.py \
    --corpus_dir ./data/procis/corpus/procis.corpus-tevatron.jsonl \
    --query_dir ./data/procis/queries/procis.train-filtered1000.queries.doct5query-100.chunk${i}.jsonl \
    --output_dir ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
    --qrels_dir ./data/procis/qrels/procis.train-filtered1000.qrels.turn-link.txt \
    --mode ${mode}
done

# run relevance prediction
for i in 0 1 2 3
do
gpuid=$((i)) 
CUDA_VISIBLE_DEVICES=${gpuid} \
nohup python -m tevatron.reranker.driver.rerank \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --dataset_path ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
  --fp16 \
  --per_device_eval_batch_size 32 \
  --rerank_max_len $(( 32 + ${doc_len} )) \
  --dataset_name json \
  --query_prefix "query: " \
  --passage_prefix "document: " \
  --rerank_output_path ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-${mode}-rankllama${doc_len}.chunk${i}.txt \
  > procis.train-filtered1000.queries.doct5query-100-${mode}-rankllama${doc_len}.chunk${i}.log 2>&1 &
done
```

#### WebDisc
Similarly, conduct query--document relevance prediction on WebDisc.
The relevance score file will be stored in `./data/webdisc/filter/`.

```bash
mode=q2d
doc_len=512

# generate relevance prediction input file
for i in 0 1 2 3
do
python prepare_rerank_file.py \
    --corpus_dir ./data/webdisc/corpus/webdisc.corpus-tevatron.jsonl \
    --query_dir ./data/webdisc/queries/webdisc.train.queries.doct5query-100.chunk${i}.jsonl \
    --output_dir ./data/webdisc/filter/webdisc.train.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
    --qrels_dir ./data/webdisc/qrels/webdisc.train.qrels.txt \
    --mode ${mode}
done

# run relevance prediction
for i in 0 1 2 3
do
gpuid=$((i)) 
CUDA_VISIBLE_DEVICES=${gpuid} \
nohup python -m tevatron.reranker.driver.rerank \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --dataset_path ./data/webdisc/filter/webdisc.train.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
  --fp16 \
  --per_device_eval_batch_size 32 \
  --rerank_max_len $(( 32 + ${doc_len} )) \
  --dataset_name json \
  --query_prefix "query: " \
  --passage_prefix "document: " \
  --rerank_output_path ./data/webdisc/filter/webdisc.train.queries.doct5query-100-${mode}-rankllama${doc_len}.chunk${i}.txt \
  > webdisc.train.queries.doct5query-100-${mode}-rankllama${doc_len}.chunk${i}.log 2>&1 &
done
```

#### 2.2.2 Query–conversation relevance <a name="222-query-conversation-relevance"></a>

#### ProCIS

Run the following commands to prepare inputs of the relevance model, and conduct query--conversation relevance prediction on ProCIS.
The relevance score file will be stored in `./data/procis/filter/`.
```bash
mode=q2C
doc_len=512

# generate relevance prediction input file
for i in 0 1 2 3
do
python prepare_rerank_file.py \
    --corpus_dir ./data/procis/corpus/procis.train-filtered1000.queries.cur.jsonl \ # we use the current user utterance representing the conversational context
    --query_dir ./data/procis/queries/procis.train-filtered1000.queries.doct5query-100.chunk${i}.jsonl \
    --output_dir ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
    --qrels_dir ./data/procis/qrels/procis.train-filtered1000.qrels.turn-link.txt \
    --mode ${mode}
done

# run relevance prediction
for i in 0 1 2 3
do
gpuid=$((i)) 
CUDA_VISIBLE_DEVICES=${gpuid} \
nohup python -m tevatron.reranker.driver.rerank \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --dataset_path ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
  --fp16 \
  --per_device_eval_batch_size 32 \
  --rerank_max_len $(( 32 + ${doc_len} )) \
  --dataset_name json \
  --query_prefix "query: " \
  --passage_prefix "document: " \
  --rerank_output_path ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-${mode}-rankllama${doc_len}.chunk${i}.txt \
  > procis.train-filtered1000.queries.doct5query-100-${mode}-rankllama${doc_len}.chunk${i}.log 2>&1 &
done
```

#### WebDisc

Similarly, conduct query--conversation relevance prediction on WebDisc.
The relevance score file will be stored in `./data/webdisc/filter/`.
```bash
mode=q2c
doc_len=512

# generate relevance prediction input file
for i in 0 1 2 3
do
python prepare_rerank_file.py \
    --corpus_dir ./data/webdisc/queries/webdisc.train.queries.cur.jsonl \ # we use the current user utterance representing the conversational context
    --query_dir ./data/webdisc/queries/webdisc.train.queries.doct5query-100.chunk${i}.jsonl \
    --output_dir ./data/webdisc/filter/webdisc.train.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
    --qrels_dir ./data/webdisc/qrels/webdisc.train.qrels.txt \
    --mode ${mode}
done

# run relevance prediction
for i in 0 1 2 3
do
gpuid=$((i)) 
CUDA_VISIBLE_DEVICES=${gpuid} \
nohup python -m tevatron.reranker.driver.rerank \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --dataset_path ./data/webdisc/filter/webdisc.train.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
  --fp16 \
  --per_device_eval_batch_size 32 \
  --rerank_max_len $(( 32 + ${doc_len} )) \
  --dataset_name json \
  --query_prefix "query: " \
  --passage_prefix "document: " \
  --rerank_output_path ./data/webdisc/filter/webdisc.train.queries.doct5query-100-${mode}-rankllama${doc_len}.chunk${i}.txt \
  > webdisc.train.queries.doct5query-100-${mode}-rankllama${doc_len}.chunk${i}.log 2>&1 &
done
```


#### 2.2.3 Optimal query selection <a name="223-optimal-query-selection"></a>

#### ProCIS

Run the following command to select the optimal query target for each conversational context, based on query--document and query-conversation relevance scores.
The selected query file will be stored in `./data/procis/queries`.
```bash
mode=q2d_q2c
doc_len=512

python query_filter.py \
--query_dir ./data/procis/queries/procis.train-filtered1000.queries.doct5query-100 \
--q2d_rerank_dir ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-q2d-rankllama${doc_len} \
--q2c_rerank_dir ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-q2c-rankllama${doc_len} \
--output_dir ./data/procis/queries/procis.train-filtered1000.queries.doct5query-100-${mode}-rankllama${doc_len}-1 \
--qrels_dir ./data/procis/qrels/procis.train-filtered1000.qrels.turn-link.txt \
--num_chunks 4 --mode ${mode}
```

#### WebDisc
Similarly, running the following command produces the selected query file stored in `./data/webdisc/queries`.
```bash
mode=q2d_q2c
doc_len=512

python query_filter.py \
--query_dir ./data/webdisc/queries/webdisc.train.queries.doct5query-100 \
--q2d_rerank_dir ./data/webdisc/filter/webdisc.train.queries.doct5query-100-q2d-rankllama${doc_len} \
--q2c_rerank_dir ./data/webdisc/filter/webdisc.train.queries.doct5query-100-q2c-rankllama${doc_len} \
--output_dir ./data/webdisc/queries/webdisc.train.queries.doct5query-100-${mode}-rankllama${doc_len}-1 \
--qrels_dir ./data/webdisc/qrels/webdisc.train.qrels.txt \
--num_chunks 4 --mode ${mode}
```

## 🚀 3. Learning to generate ad-hoc queries from conversations (training)<a name="3-learning-to-generate-ad-hoc-queries-from-conversations-training"></a>

We fine-tune an LLM to learn the mapping raw conversational context to its optimal ad-hoc query target.
We use DeepSpeed to enable multi-GPU training.
We define the `his_cur2query` and `his2query` prompts, which are corresponding to the `conversation contextualisation` and `interest anticipation` settings defined in the paper, respectively.
Specifically, `his_cur2query` aims to generate ad-hoc queries based on conversational history as well as the current user utterance, while `his2query` aims to generate ad-hoc queries based on only conversational history.

#### ProCIS
Run the following commands to fine-tune an LLM to learn the mapping raw conversational context to its optimal ad-hoc query target on ProCIS.
Use `output_dir` to specify the directory where the checkpoints will be saved.
```bash
llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"

# for the conversation contextualisation setting
nohup \
deepspeed --include localhost:0,1,2,3 --master_port 60000 conv2query.py \
--model_name_or_path ${llm}$ \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--history_dir ./data/procis/queries/procis.train-filtered1000.queries.his.tsv \
--current_dir ./data/procis/queries/procis.train-filtered1000.queries.cur.tsv \
--query_dir ./data/procis/queries/procis.train-filtered1000.queries.doct5query-100-q2d_q2c-rankllama512-1-raw.tsv \
--output_dir ./data/procis/queries/ \
--checkpoint_dir ./checkpoint/ \
--logging_steps 10 \
--batch_size 8 \
--gradient_accumulation_steps 4 \
--save_steps 1000 \
--num_epochs 1.0 \
--deepspeed_config ./deepspeed/ds_zero1_config.json \
--prompt his_cur2query \
> procis.train-filtered1000.queries.his_cur2query--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw.log 2>&1 &

# for the interest anticipation setting
nohup \
deepspeed --include localhost:0,1,2,3 --master_port 60001 conv2query.py \
--model_name_or_path ${llm}$ \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--history_dir ./data/procis/queries/procis.train-filtered1000.queries.his.tsv \
--current_dir ./data/procis/queries/procis.train-filtered1000.queries.cur.tsv \
--query_dir ./data/procis/queries/procis.train-filtered1000.queries.doct5query-100-q2d_q2c-rankllama512-1-raw.tsv \
--output_dir ./data/procis/queries/ \
--checkpoint_dir ./checkpoint/ \
--logging_steps 10 \
--batch_size 8 \
--gradient_accumulation_steps 4 \
--save_steps 1000 \
--num_epochs 1.0 \
--deepspeed_config ./deepspeed/ds_zero1_config.json \
--prompt his2query \
> procis.train-filtered1000.queries.his2query--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw.log 2>&1 &
```

#### WebDisc

Similar operations are performed on WebDisc.
```bash
llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"

# for the conversation contextualisation setting
nohup \
deepspeed --include localhost:0,1,2,3 --master_port 60000 conv2query.py \
--model_name_or_path ${LLM}$ \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--history_dir ./data/webdisc/queries/webdisc.train.queries.his.tsv \
--current_dir ./data/webdisc/queries/webdisc.train.queries.cur.tsv \
--query_dir ./data/webdisc/queries/webdisc.train.queries.doct5query-100-q2d_q2c-rankllama512-1-raw.tsv \
--output_dir ./data/webdisc/queries/ \
--checkpoint_dir ./checkpoint/ \
--logging_steps 10 \
--batch_size 8 \
--gradient_accumulation_steps 4 \
--save_steps 1000 \
--num_epochs 1.0 \
--deepspeed_config ./deepspeed/ds_zero1_config.json \
--prompt his_cur2query \
> webdisc.train.queries.his_cur2query--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw.log 2>&1 &

# for the interest anticipation setting
nohup \
deepspeed --include localhost:0,1,2,3 --master_port 60001 conv2query.py \
--model_name_or_path ${llm}$ \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--history_dir ./data/webdisc/queries/webdisc.train.queries.his.tsv \
--current_dir ./data/webdisc/queries/webdisc.train.queries.cur.tsv \
--query_dir ./data/webdisc/queries/webdisc.train.queries.doct5query-100-q2d_q2c-rankllama512-1-raw.tsv \
--output_dir ./data/webdisc/queries/ \
--checkpoint_dir ./checkpoint/ \
--logging_steps 10 \
--batch_size 8 \
--gradient_accumulation_steps 4 \
--save_steps 1000 \
--num_epochs 1.0 \
--deepspeed_config ./deepspeed/ds_zero1_config.json \
--prompt his2query \
> webdisc.train.queries.his2query--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw.log 2>&1 &
```

## 🔎 4. Generating ad-hoc queries for retrieval (inference) <a name="4-generating-ad-hoc-queries-for-retrieval-inference"></a>

#### ProCIS

At test, run the following commands to generate ad-hoc queries for conversational contexts under the two settings on the `dev`, `future_dev`, and `test` sets of ProCIS.
Use `output_dir` to specify the directory where the generated queries will be saved.
```bash
llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
prompt=his_cur2query
ckpt=procis.train-filtered1000.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=4751
gpuid=0


# for the conversation contextualisation setting
for p in his_cur2query his2query
do
for s in dev future_dev test
do
CUDA_VISIBLE_DEVICES=${gpuid} python conv2query.py \
--model_name_or_path ${llm} \
--checkpoint_name ${ckpt}/checkpoint-${step} \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--history_dir ./data/procis/queries/procis.${s}.queries.his.tsv \
--current_dir ./data/procis/queries/procis.${s}.queries.cur.tsv \
--output_dir ./data/procis/queries/ \
--checkpoint_dir ./checkpoint/ \
--batch_size 16  \
--logging_steps 10 \
--prompt ${prompt} \
--infer --verbose
done
done

llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
prompt=his2query
ckpt=procis.train-filtered1000.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=4751
gpuid=0

# for the interest anticipation setting
for s in dev future_dev test
do
CUDA_VISIBLE_DEVICES=${gpuid} python conv2query.py \
--model_name_or_path ${llm} \
--checkpoint_name ${ckpt}/checkpoint-${step} \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--history_dir ./data/procis/queries/procis.${s}.queries.his.tsv \
--current_dir ./data/procis/queries/procis.${s}.queries.cur.tsv \
--output_dir ./data/procis/queries/ \
--checkpoint_dir ./checkpoint/ \
--batch_size 16  \
--logging_steps 10 \
--prompt ${prompt} \
--infer --verbose
done
```

#### WebDisc
At test, run the following commands to generate ad-hoc queries for conversational contexts under the two settings on the `val` and `test` sets of WebDisc.
Use `output_dir` to specify the directory where the generated queries will be saved.
```bash
# for the conversation contextualisation setting
prompt=his_cur2query
llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
ckpt=webdisc.train.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=1003
gpuid=0

for s in dev future_dev test
do
CUDA_VISIBLE_DEVICES=${gpuid} python conv2query.py \
--model_name_or_path ${llm} \
--checkpoint_name ${ckpt}/checkpoint-${step} \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--history_dir ./data/webdisc/queries/webdisc.${s}.queries.his.tsv \
--current_dir ./data/webdisc/queries/webdisc.${s}.queries.cur.tsv \
--output_dir ./data/webdisc/queries/ \
--checkpoint_dir ./checkpoint/ \
--batch_size 16  \
--logging_steps 10 \
--prompt ${prompt} \
--infer --verbose
done

# for the interest anticipation setting
prompt=his2query
llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
ckpt=webdisc.train.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=1003
gpuid=0

for s in val test
do
CUDA_VISIBLE_DEVICES=${gpuid} python conv2query.py \
--model_name_or_path ${llm} \
--checkpoint_name ${ckpt}/checkpoint-${step} \
--token ${TOKEN} \
--cache_dir ${CACHE_DIR} \
--history_dir ./data/webdisc/queries/webdisc.${s}.queries.his.tsv \
--current_dir ./data/webdisc/queries/webdisc.${s}.queries.cur.tsv \
--output_dir ./data/webdisc/queries/ \
--checkpoint_dir ./checkpoint/ \
--batch_size 16  \
--logging_steps 10 \
--prompt ${prompt} \
--infer --verbose
done
```

## 5. Reusing off-the-shelf ad-hoc retrievers <a name="5-reusing-off-the-shelf-ad-hoc-retrievers"></a>

We use BM25, ANCE, SPLADE++ and RepLLaMA as off-the-shelf retrievers.
We use BM25 and ANCE from [Pyserini](https://github.com/castorini/pyserini); we use SPLADE++ from the official repository of [SPLADE](https://github.com/naver/splade); and we use RepLLaMA from [Tevatron](https://github.com/texttron/tevatron).
Note that ANCE, SPLADE++ and RepLLaMA have been pre-trained on the training set of MS MARCO V1 (passage retrieval).

In the following part, we show an example of reusing BM25, ANCE via [Pyserini](https://github.com/castorini/pyserini), as well as RepLLaMA via [Tevatron](https://github.com/texttron/tevatron) under the `conversation contextualisation` and `interest anticipation` settings.
Please follow the instruction in [SPLADE](https://github.com/naver/splade) to reuse SPLADE++.


### 5.1 BM25/ANCE indexing

#### ProCIS

Run the following commands to index the ProCIS corpus for BM25 and ANCE retrieval:
```bash
# bm25
python -m pyserini.index.lucene \
--collection JsonCollection \
--input ./data/procis/corpus/procis.corpus.jsonl \
--index ./data/procis/indexs/procis.index.bm25 \
--generator DefaultLuceneDocumentGenerator \
--threads 16 \
--storePositions --storeDocvectors --storeRaw

# ance
nohup \
python -m pyserini.encode \
  input   --corpus ./data/procis/corpus/procis.corpus.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ./data/procis/indexes/procis.index.ance-msmarco-passage \
          --to-faiss \
  encoder --encoder castorini/ance-msmarco-passage \
          --fields text \
          --device cuda:0 \
          --batch 256 \
          --max-length 256 \
          > procis.index.ance-msmarco-passage.log 2>&1 &

```

#### WebDisc

Run the following commands to index the WebDisc corpus for BM25 and ANCE retrieval:
```bash
python -m pyserini.index.lucene \
--collection JsonCollection \
--input ./data/webdisc/corpus/webdisc.corpus.jsonl \
--index ./data/webdisc/indexes/webdisc.index.bm25 \
--generator DefaultLuceneDocumentGenerator \
--threads 16 \
--storePositions --storeDocvectors --storeRaw

nohup \
python -m pyserini.encode \
  input   --corpus ./data/webdisc/corpus/webdisc.corpus.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ./data/webdisc/indexes/webdisc.index.ance-msmarco-passage \
          --to-faiss \
  encoder --encoder castorini/ance-msmarco-passage \
          --fields text \
          --device cuda:0 \
          --batch 128 \
          --max-length 512 \
          > webdisc.index.ance-msmarco-passage.log 2>&1 &
```

### 5.2 RepLLaMA indexing

Run the following commands to index the ProCIS corpus for RepLLaMA retrieval:
#### ProCIS
```bash
q_len=64
psg_len=256
mkdir ./data/webdisc/indexes/procis.index.psg${psg_len}--repllama-v1-7b-lora-passage

for s in 0 1 2 3
do
gpuid=${s} \
CUDA_VISIBLE_DEVICES=${s} \
nohup \
python -m tevatron.retriever.driver.encode \
  --output_dir=./temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
  --lora \
  --query_prefix "query:" \
  --passage_prefix "passage:" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 64 \
  --query_max_len ${q_len} \
  --passage_max_len ${psg_len} \
  --dataset_path ./data/procis/corpus/procis.corpus-tevatron.jsonl \
  --dataset_config jsonl \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} \
  --encode_output_path ./data/procis/indexes/procis.index.psg256--repllama-v1-7b-lora-passage/${s}.pkl \
  > procis.index.psg256--repllama-v1-7b-lora-passage.${s}.log 2>&1 &
done

```

#### WebDisc
similarly, run the following commands to index the WebDisc corpus for RepLLaMA retrieval:
```bash
q_len=64
psg_len=512
mkdir ./data/webdisc/indexes/webdisc.index.psg${psg_len}--repllama-v1-7b-lora-passage


for i in 0 1 2 3
do
CUDA_VISIBLE_DEVICES=$((i)) \
nohup \
python -m tevatron.retriever.driver.encode \
  --output_dir=./temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
  --lora \
  --query_prefix "query:" \
  --passage_prefix "passage:" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 32 \
  --query_max_len ${q_len} \
  --passage_max_len ${psg_len} \
  --dataset_path ./data/webdisc/corpus/webdisc.corpus-tevatron.jsonl \
  --dataset_config jsonl \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${i} \
  --encode_output_path ./data/webdisc/indexes/webdisc.index.psg512--repllama-v1-7b-lora-passage/${i}.pkl \
  > webdisc.index.psg512--repllama-v1-7b-lora-passage.${i}.log 2>&1 &
done 

```

### 5.3 BM25/ANCE retrieval and evaluation
#### ProCIS

Run the following commands to perform BM25/ANCE retrieval on the dev, future dev and test sets of ProCIS under the two settings:
```bash

# bm25
for prompt in his_cur2query his2query
do

llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
ckpt=procis.train-filtered1000.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=4751
q=${prompt}--${llm_short}--ckpt-${ckpt}-step-${step}
k1=0.9
b=0.4

for s in dev future_dev test
do
python -m pyserini.search.lucene \
 --topics ./data/procis/queries/procis.${s}.queries.${q}.tsv \
 --index ./data/procis/indexes/procis.index.bm25 \
 --output ./data/procis/runs/procis.${s}.run.${q}--bm25-k1-${k1}-b-${b}.txt \
 --bm25 --hits 1000 --batch-size 512 --k1 ${k1} --b ${b}


# evaluation
    if [ "$s" = "test" ]; then
      qrels_file="./data/procis/qrels/procis.${s}.qrels.turn-manual.txt"
    else
      qrels_file="./data/procis/qrels/procis.${s}.qrels.turn-link.txt"
    fi

echo ${s} ${q} bm25-k1-${k1}-b-${b}  
python -u evaluate_ranking.py \
--run_dir ./data/procis/runs/procis.${s}.run.${q}--bm25-k1-${k1}-b-${b}.txt \
--qrels_dir ${qrels_file} \
--rel_scale 1

done
done


# ance
gpuid=0

for prompt in his_cur2query his2query
do

llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
ckpt=procis.train-filtered1000.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=4751
q=${prompt}--${llm_short}--ckpt-${ckpt}-step-${step}

for s in dev future_dev test
do
python -m pyserini.search.faiss \
 --threads 16 --batch-size 512 --hits 1000 --device cuda:${gpuid} \
 --index ./data/procis/indexes/procis.index.ance-msmarco-passage \
 --topics ./data/procis/queries/procis.${s}.queries.${q}.tsv \
 --encoder castorini/ance-msmarco-passage \
 --output ./data/procis/runs/procis.${s}.run.${q}--ance-msmarco-passage.txt


# evaluation
    if [ "$s" = "test" ]; then
      qrels_file="./data/procis/qrels/procis.${s}.qrels.turn-manual.txt"
    else
      qrels_file="./data/procis/qrels/procis.${s}.qrels.turn-link.txt"
    fi

echo ${s} ${q} ance-msmarco-passage 
python -u evaluate_ranking.py \
--run_dir ./data/procis/runs/procis.${s}.run.${q}--ance-msmarco-passage.txt \
--qrels_dir ${qrels_file} \
--rel_scale 1

done
done

```

#### WebDisc

Run the following commands to perform BM25/ANCE retrieval on the test and val sets of WebDisc under the two settings:
```bash
# bm25
for prompt in his_cur2query his2query
do

llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
ckpt=webdisc.train.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=1003
q=${prompt}--${llm_short}--ckpt-${ckpt}-step-${step}
k1=4
b=0.9

for s in val test
do
python -m pyserini.search.lucene \
 --stopwords ./data/webdisc/raw/stopwords.txt \ # follow the original authors to consider stop words.
 --topics ./data/webdisc/queries/webdisc.${s}.queries.${q}.tsv \
 --index ./data/webdisc/indexes/webdisc.index.bm25 \
 --output ./data/webdisc/runs/webdisc.${s}.run.${q}--bm25-k1-${k1}-b-${b}_remove_stopwords.txt \
 --bm25 --hits 1000 --batch-size 512 --k1 ${k1} --b ${b}


# evaluation
echo ${s} ${q} bm25-k1-${k1}-b-${b}_remove_stopwords
python -u evaluate_ranking.py \
--run_dir ./data/webdisc/runs/webdisc.${s}.run.${q}--bm25-k1-${k1}-b-${b}_remove_stopwords.txt \
--qrels_dir ./data/webdisc/qrels/webdisc.${s}.qrels.txt \
--rel_scale 1

done
done


# ance
gpuid=0 

for prompt in his_cur2query his2query
do

llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
ckpt=webdisc.train.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=1003
q=${prompt}--${llm_short}--ckpt-${ckpt}-step-${step}

for s in val test 
do
python -m pyserini.search.faiss \
 --threads 16 --batch-size 512 --hits 1000 --max-length 512 --device cuda:${gpuid} \
 --index ./data/webdisc/indexes/webdisc.index.ance-msmarco-passage \
 --topics ./data/webdisc/queries/webdisc.${s}.q.${q}-checkpoint-${ckpt}.tsv \
 --encoder castorini/ance-msmarco-passage \
 --output ./data/webdisc/runs/webdisc.${s}.run.${q}-checkpoint-${ckpt}--ance-msmarco-passage.txt

# evaluation
echo ${s} ${q} ance-msmarco-passage
python -u evaluate_ranking.py \
--run_dir ./data/webdisc/runs/webdisc.${s}.run.${q}--ance-msmarco-passage.txt \
--qrels_dir ./data/webdisc/qrels/webdisc.${s}.qrels.txt \
--rel_scale 1
done
done

```

### 5.4 RepLLaMA retrieval and evaluation
#### ProCIS
```bash

gpuid=0

for prompt in his_cur2query his2query
do

llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
ckpt=procis.train-filtered1000.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=4751
q=${prompt}--${llm_short}--ckpt-${ckpt}-step-${step}


for s in dev future_dev test
do

q_len=64
psg_len=256
run=procis.${s}.run.${q}-${q_len}-psg${psg_len}--repllama-v1-7b-lora-passage.gpu
mkdir ./data/procis/runs/${run}_

# query encoding
CUDA_VISIBLE_DEVICES=${gpuid} \
python -m tevatron.retriever.driver.encode \
  --output_dir=./temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
  --lora \
  --query_prefix "query:" \
  --passage_prefix "passage:" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len ${q_len} \
  --passage_max_len ${psg_len} \
  --dataset_path ./data/procis/queries/procis.${s}.queries.${q}.jsonl \
  --dataset_config jsonl \
  --encode_output_path ./data/procis/queries/procis.${s}.queries.${q}-${q_len}--repllama-v1-7b-lora-passage.pkl

# search
for shard in 0 1 2 3
do
CUDA_VISIBLE_DEVICES=${gpuid} \
python -m tevatron.retriever.driver.search \
    --query_reps ./data/procis/queries/procis.${s}.queries.${q}-${q_len}--repllama-v1-7b-lora-passage.pkl \
    --passage_reps ./data/procis/indexes/procis.index.psg${psg_len}--repllama-v1-7b-lora-passage/${shard}.pkl \
    --depth 1000 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to ./data/procis/runs/${run}_/${shard}.txt
done

python -m tevatron.scripts.reduce_results \
--results_dir ./data/procis/runs/${run}_ \
--output ./data/procis/runs/${run}_.txt \
--depth 1000

# convert to trec format
python -m tevatron.utils.format.convert_result_to_trec \
              --input ./data/procis/runs/${run}_.txt \
              --output ./data/procis/runs/${run}.txt


# evaluation
    if [ "$s" = "test" ]; then
      qrels_file="./data/procis/qrels/procis.${s}.qrels.turn-manual.txt"
    else
      qrels_file="./data/procis/qrels/procis.${s}.qrels.turn-link.txt"
    fi
    
echo ${run} ${qrels_file}
python -u evaluate_ranking.py \
--run_dir ./data/procis/runs/${run}.txt \
--qrels_dir ${qrels_file} \
--rel_scale 1

done
done
```

#### WebDisc
```bash

gpuid=0 

for prompt in his_cur2query his2query
do

llm="mistralai/Mistral-7B-Instruct-v0.3"
llm_short="${llm##*/}"
ckpt=webdisc.train.queries.${prompt}--${llm_short}--doct5query-100-q2d_q2c-rankllama512-1-raw
step=1003
q=${prompt}--${llm_short}--ckpt-${ckpt}-step-${step}


for s in val test
do

q_len=64
psg_len=512
run=webdisc.${s}.run.${q}-${q_len}-psg${psg_len}--repllama-v1-7b-lora-passage.gpu 
mkdir ./data/webdisc/runs/${run}_

# query encoding
CUDA_VISIBLE_DEVICES=${gpuid} \
python -m tevatron.retriever.driver.encode \
  --output_dir=./temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
  --lora \
  --query_prefix "query:" \
  --passage_prefix "passage:" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 32 \
  --query_max_len ${q_len} \
  --passage_max_len ${psg_len} \
  --dataset_path ./data/webdisc/queries/webdisc.${s}.queries.${q}.jsonl \
  --dataset_config jsonl \
  --encode_output_path ./data/webdisc/queries/webdisc.${s}.queries.${q}-${q_len}--repllama-v1-7b-lora-passage.pkl

# search
for shard in 0 1 2 3
do
CUDA_VISIBLE_DEVICES=${gpuid} \
python -m tevatron.retriever.driver.search \
    --query_reps ./data/webdisc/queries/webdisc.${s}.queries.${q}-${q_len}--repllama-v1-7b-lora-passage.pkl \
    --passage_reps ./data/webdisc/indexes/webdisc.index.psg${psg_len}--repllama-v1-7b-lora-passage/${shard}.pkl \
    --depth 1000 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to ./data/webdisc/runs/${run}_/${shard}.txt
done


python -m tevatron.scripts.reduce_results \
--results_dir ./data/webdisc/runs/${run}_ \
--output ./data/webdisc/runs/${run}_.txt \
--depth 1000

# convert to trec format
python -m tevatron.utils.format.convert_result_to_trec \
              --input ./data/webdisc/runs/${run}_.txt \
              --output ./data/webdisc/runs/${run}.txt
              
# evaluation
echo ${run} 
python -u evaluate_ranking.py \
--run_dir ./data/webdisc/runs/${run}.txt \
--qrels_dir ./data/webdisc/qrels/webdisc.${s}.qrels.txt \
--rel_scale 1

done
done
```

## 🎨 6. Further fine-tuning ad-hoc retrievers using filtered ad-hoc queries (optional) <a name="6-further-fine-tuning-ad-hoc-retrievers-using-filtered-ad-hoc-queries-optional"></a>
In this section we show an example of fine-tuning RepLLaMA via [Tevatron](https://github.com/texttron/tevatron) using our generated ad-hoc queries.
Please follow the official repositories of [SPLADE++](https://github.com/naver/splade) and [ANCE](https://github.com/microsoft/ANCE) to fine-tune them.

We follow the negative sampling process of RepLLaMA to get hard negatives.

After fine-tuning, please follow Section [Reusing off-the-shelf ad-hoc retrievers](#5-reusing-off-the-shelf-ad-hoc-retrievers-training) to do indexing and retrieval using the further fine-tuned checkpoints.

### 6.1 Negative generation

#### ProCIS
Run the following commands to obtain BM25 result lists for (1) conversational history only and (2) conversational history plus the current user utterance on the training set of ProCIS.
Then, sample hard negatives and generate the final training data stored in the `./data/procis/training/`:
```bash
# get BM25 result lists
k1=0.9
b=0.4
nohup \
python -m pyserini.search.lucene \
 --topics ./data/procis/queries/procis.train-filtered1000.queries.his-cur.tsv \
 --index ./data/procis/indexes/procis.index.bm25 \
 --output ./data/procis/runs/procis.train-filtered1000.run.his-cur--bm25-k1-${k1}-b-${b}.txt \
 --bm25 --hits 200 --k1 ${k1} --b ${b} --threads 16 --batch-size 128 \
> procis.train-filtered1000.his-cur--bm25.log 2>&1 &

nohup \
python -m pyserini.search.lucene \
 --topics ./data/procis/queries/procis.train-filtered1000.queries.his.tsv \
 --index ./data/procis/indexes/procis.index.bm25 \
 --output ./data/procis/runs/procis.train-filtered1000.run.his--bm25-k1-${k1}-b-${b}.txt \
 --bm25 --hits 200 --k1 ${k1} --b ${b} --threads 16 --batch-size 128 \
> procis.train-filtered1000.his--bm25.log 2>&1 &

# get BM25's hard negatives and generate the final training data
python -u preprocess_retriever_training.py \
--corpus_dir ./data/procis/corpus/procis.corpus.jsonl/procis.corpus.jsonl \
--query_dir ./data/procis/queries/procis.train-filtered1000.doct5query-100-q2d_q2c-rankllama512-1-concat.tsv \
--qrels_dir ./data/procis/qrels/procis.train-filtered1000.qrels.turn-link.txt \
--run1_dir ./data/procis/runs/procis.train-filtered1000.run.his--bm25-k1-${k1}-b-${b}.txt \
--run2_dir ./data/procis/runs/procis.train-filtered1000.run.his-cur--bm25-k1-${k1}-b-${b}.txt \
--output_dir ./data/procis/training/
```

#### WebDisc
Similarly, run the following commands to get BM25 result lists, sample hard negatives, and generate the final training data for WebDisc.
The final generated training data will be stored in`./data/webdisc/training/`:
```bash
# get BM25 result lists
k1=8
b=0.99
for q in his-cur
do
for s in train
do
python -m pyserini.search.lucene \
 --stopwords ./data/webdisc/raw/stopwords.txt \
 --topics ./data/webdisc/queries/webdisc.${s}.queries.${q}.tsv \
 --index ./data/webdisc/indexes/webdisc.index.bm25 \
 --output ./data/webdisc/runs/webdisc.${s}.run.${q}--bm25-k1-${k1}-b-${b}_remove_stopwords.txt \
 --bm25 --hits 1000 --k1 ${k1} --b ${b} --threads 16 --batch-size 64
done
done 

k1=7
b=0.99
for q in his
do
for s in train
do
python -m pyserini.search.lucene \
 --stopwords ./data/webdisc/raw/stopwords.txt \
 --topics ./data/webdisc/queries/webdisc.${s}.queries.${q}.tsv \
 --index ./data/webdisc/indexes/webdisc.index.bm25 \
 --output ./data/webdisc/runs/webdisc.${s}.run.${q}--bm25-k1-${k1}-b-${b}_remove_stopwords.txt \
 --bm25 --hits 1000 --k1 ${k1} --b ${b} --threads 16 --batch-size 64
done
done

# get BM25's hard negatives
python -u preprocess_retriever_training.py \
--corpus_dir ./data/webdisc/corpus/webdisc.corpus-tevatron.jsonl \
--query_dir ./data/webdisc/queries/webdisc.train.doct5query-100-q2d_q2c-rankllama512-1-concat.tsv \
--qrels_dir ./data/webdisc/qrels/webdisc.train.qrels.txt \
--run1_dir ./data/webdisc/runs/webdisc.train.run.his--bm25-k1-7-b-0.99_remove_stopwords.txt \
--run2_dir ./data/webdisc/runs/webdisc.train.run.his-cur--bm25-k1-8-b-0.99_remove_stopwords.txt \
--output_dir ./data/webdisc/training/
```

### 6.2 Further fine-tuning RepLLaMA

Please run the following command to further fine-tune RepLLaMA using our generated ad-hoc queries on the training set of ProCIS:
#### ProCIS
```bash
q_len=64
psg_len=256

nohup \
deepspeed --include localhost:0,1,2,3 --master_port 60000 \
--module tevatron.retriever.driver.train \
  --deepspeed ./deepspeed/ds_zero3_config.json \
  --output_dir ./checkpoints/procis.train-filtered1000.doct5query-100-q2d_q2c-rankllama512-1-concat${q_len}-psg${psg_len}-Llama-2-7b-hf-repllama-v1-7b-lora-passage--neg20 \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \ # make sure RepLLaMA has been initialised by the checkpoint pre-trained on MS MARCO
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 50 \
  --dataset_name json \
  --dataset_path ./data/procis/training/procis.train-filtered1000.doct5query-100-q2d_q2c-rankllama512-1-concat--neg20.jsonl \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len ${q_len} \
  --passage_max_len ${psg_len} \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --lora_r 32 \
  > procis.train-filtered1000.doct5query-100-q2d_q2c-rankllama512-1-concat${q_len}-psg${psg_len}-Llama-2-7b-hf-repllama-v1-7b-lora-passage--neg20.log 2>&1 &
```

#### WebDisc
Please run the following command to further fine-tune RepLLaMA using our generated ad-hoc queries on the training set of WebDisc:
```bash
q_len=64
psg_len=512

nohup \
deepspeed --include localhost:0,1,2,3 --master_port 60001 \
--module tevatron.retriever.driver.train \
  --deepspeed ./deepspeed/ds_zero3_config.json \
  --output_dir ./checkpoints/webdisc.train.doct5query-100-q2d_q2c-rankllama512-1-concat64-psg256-Llama-2-7b-hf-repllama-v1-7b-lora-passage--neg20 \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \ # make sure RepLLaMA has been initialised by the checkpoint pre-trained on MS MARCO
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 50 \
  --dataset_name json \
  --dataset_path ./data/webdisc/training/webdisc.train.doct5query-100-q2d_q2c-rankllama512-1-concat--neg20.jsonl \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len ${q_len} \
  --passage_max_len ${psg_len} \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --lora_r 32 \
  > webdisc.train.doct5query-100-q2d_q2c-rankllama512-1-concat${q_len}-psg${psg_len}-Llama-2-7b-hf-repllama-v1-7b-lora-passage--neg20.log 2>&1 &
```
