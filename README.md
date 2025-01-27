# Conv2Query

This is the repository for the paper titled "Bridging the Gap: From Ad-hoc to Proactive Search in Conversations".

This repository is structured into the following parts:
1. Prerequisite
   * 1.1 Install dependencies
   * 1.2 Data preparation
2. Conv2Query



## ⚙️ 1. Prerequisite

## 1.1 Install dependencies
```bash
pip install -r requirements.txt
```

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
unzip ./data/procis/raw/procis.zip -d ./data/procis/raw/
mv ./data/procis/raw/procis/procis/collection.jsonl ./data/procis/raw/collection.jsonl
mv ./data/procis/raw/procis/procis/dev.jsonl ./data/procis/raw/dev.jsonl
mv ./data/procis/raw/procis/procis/future_dev.jsonl ./data/procis/raw/future_dev.jsonl
mv ./data/procis/raw/procis/procis/test.jsonl ./data/procis/raw/test.jsonl
mv ./data/procis/raw/procis/procis/train.jsonl ./data/procis/raw/train.jsonl
```

Next, execute the script to preprocess the ProCIS dataset:
```bash
python -u ./preprocess_procis.py
```

### The [WebDisc](https://dl.acm.org/doi/10.1145/3578337.3605139) dataset  (published at ICTIR 2023)
Please ask the original authors of WebDisc to get the raw data, and then put the data in the directory of `./data/webdisc/raw`. After decompressing, execute the script to preprocess the ProCIS dataset:
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

```bash
python -u ./preprocess_webdisc.py
```


## 2. 📜 Generating ad-hoc queries from documents


## 3. 🔬 Query filtering based on document relevance and conversation alignment (QF-DC)


## 4. 🚀 Learning to generate ad-hoc queries from conversations


## 5. 🔎 Generating ad-hoc queries for retrieval (inference)


## 6. 🎨 Further fine-tuning ad-hoc retrievers using filtered ad-hoc queries (Optional)






