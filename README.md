# Conv2Query

This is the repository for the paper titled **Bridging the Gap: From Ad-hoc to Proactive Search in Conversations**.

This repository is structured into the following parts:
1. Prerequisite
   * 1.1 Install dependencies
   * 1.2 Data preparation
2. Producing pseudo ad-hoc query targets for training
   * 2.1 Generating ad-hoc queries from documents
   * 2.2 Query filtering based on document relevance and conversation alignment (QF-DC)
3. Learning to generate ad-hoc queries from conversations
4. Generating ad-hoc queries for retrieval (inference)
5. Further fine-tuning ad-hoc retrievers using filtered ad-hoc queries (optional)



## ⚙️ 1. Prerequisite

## 1.1 Install dependencies
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

## 1.2 Data preparation 

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


## 📜 2. Producing pseudo ad-hoc query targets for training

## 2.1 Generating ad-hoc queries from documents

### ProCIS
Please use the following commands to run [Doc2Query-T5](https://huggingface.co/BeIR/query-gen-msmarco-t5-large-v1) to generate 100 ad-hoc queries per relevant document for each conversational context.
Alternatively, we provide the script to run [Doc2Query-Llama2](https://huggingface.co/soyuj/llama2-doc2query) to generate 70 queries per relevant document; we set the number of query to 70 because the GPU memory limitation. 
```bash
# Doc2Query-T5
for i in 0 1 2 3
do
gpu_id=$((i)) 
CUDA_VISIBLE_DEVICES=${gpu_id} \
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
gpu_id=$((i)) 
CUDA_VISIBLE_DEVICES=${gpu_id} \
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
The generated queries will be stored in `data/procis/queries`.

### WebDisc
The following operations are similar to ProCIS:


```bash
# Doc2Query-T5
for i in 0 1 2 3
do
gpu_id=$((i)) 
CUDA_VISIBLE_DEVICES=${gpu_id} \
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
gpu_id=$((i)) 
CUDA_VISIBLE_DEVICES=${gpu_id} \
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
The generated queries will be stored in `data/webdisc/queries`.

## 2.2. 🔬 Query filtering based on document relevance and conversation alignment (QF-DC)
For predicting query--document relevance and query--conversation relevance, we use [RepLLaMA](https://huggingface.co/castorini/repllama-v1-7b-lora-passage) as our relevance model. We use the Tevatron package.

### 2.2.1 Query--document relevance

#### ProCIS

Run the following commands to prepare inputs of the relevance model, and conduct query--document relevance prediction on ProCIS:
```bash
mode=q2d
doc_len=512

# generate re-ranking input file
for i in 0 1 2 3
do
python prepare_rerank_file.py \
    --corpus_dir ./data/procis/corpus/procis.corpus-tevatron.jsonl \
    --query_dir ./data/procis/queries/procis.train-filtered1000.queries.doct5query-100.chunk${i}.jsonl \
    --output_dir ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
    --qrels_dir ./data/procis/qrels/procis.train-filtered1000.qrels.turn-link.txt \
    --mode ${mode}
done

# run re-ranking
for i in 0 1 2 3
do
gpu_id=$((i)) 
CUDA_VISIBLE_DEVICES=${gpu_id} \
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
Similarly, conduct query--document relevance prediction on WebDisc:

```bash
mode=q2d
doc_len=512

# generate re-ranking input file
for i in 0 1 2 3
do
python prepare_rerank_file.py \
    --corpus_dir ./data/webdisc/corpus/webdisc.corpus-tevatron.jsonl \
    --query_dir ./data/webdisc/queries/webdisc.train.queries.doct5query-100.chunk${i}.jsonl \
    --output_dir ./data/webdisc/filter/webdisc.train.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
    --qrels_dir ./data/webdisc/qrels/webdisc.train.qrels.txt \
    --mode ${mode}
done

# run re-ranking
for i in 0 1 2 3
do
gpu_id=$((i)) 
CUDA_VISIBLE_DEVICES=${gpu_id} \
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

### 2.2.2 Query--conversation relevance

#### ProCIS

Run the following commands to prepare inputs of the relevance model, and conduct query--conversation relevance prediction on ProCIS:
```bash
mode=q2C
doc_len=512

# generate re-ranking input file
for i in 0 1 2 3
do
python prepare_rerank_file.py \
    --corpus_dir ./data/procis/corpus/procis.train-filtered1000.queries.cur.jsonl \
    --query_dir ./data/procis/queries/procis.train-filtered1000.queries.doct5query-100.chunk${i}.jsonl \
    --output_dir ./data/procis/filter/procis.train-filtered1000.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
    --qrels_dir ./data/procis/qrels/procis.train-filtered1000.qrels.turn-link.txt \
    --mode ${mode}
done

# run re-ranking
for i in 0 1 2 3
do
gpu_id=$((i)) 
CUDA_VISIBLE_DEVICES=${gpu_id} \
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

Similarly, conduct query--conversation relevance prediction on WebDisc:
```bash
mode=q2c
doc_len=512

# generate re-ranking input file
for i in 0 1 2 3
do
python prepare_rerank_file.py \
    --corpus_dir ./data/webdisc/queries/webdisc.train.queries.cur.jsonl \
    --query_dir ./data/webdisc/queries/webdisc.train.queries.doct5query-100.chunk${i}.jsonl \
    --output_dir ./data/webdisc/filter/webdisc.train.queries.doct5query-100-${mode}-rank_input.chunk${i}.jsonl \
    --qrels_dir ./data/webdisc/qrels/webdisc.train.qrels.txt \
    --mode ${mode}
done

# run re-ranking
for i in 0 1 2 3
do
gpu_id=$((i)) 
CUDA_VISIBLE_DEVICES=${gpu_id} \
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


### 2.2.3 Score aggregation

#### ProCIS
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



## 3. 🚀 Learning to generate ad-hoc queries from conversations


## 4. 🔎 Generating ad-hoc queries for retrieval (inference)


## 5. 🎨 Further fine-tuning ad-hoc retrievers using filtered ad-hoc queries (Optional)






