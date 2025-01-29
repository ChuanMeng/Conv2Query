import sys
sys.path.append('./')
import json
from tqdm import tqdm
import logging
import os
import pytrec_eval
from collections import defaultdict


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def limit_str_tokens(text, limit):
    tokens = text.split(" ")
    return " ".join(tokens[:limit])

def truncate_left_tokens(text, limit):
    tokens = text.split(" ")
    return " ".join(tokens[-limit:])

def load_queries(path):
    queries = {}

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            split_line = line.split('\t')
            qid = split_line[0]
            query = " ".join(split_line[1:])

            query = " | ".join(query.split('<C>')).strip()

            if query == "":
                query = "[link]"

            query = truncate_left_tokens(query, 512).strip()

            queries[qid] = query

    return queries

def load_webpages(path):
    webpages = {}
    for file in os.listdir(path):
        webpage = json.load(open(path + file, 'r'))
        webpages[webpage['id']] = webpage['contents']
    return webpages

def preprocess_webdisc():
    # collection

    logging.info("Preprocessing corpus...")

    count=0

    webpages = load_webpages("./data/webdisc/raw/webpages/")

    with open("./data/webdisc/corpus/webdisc.corpus.jsonl/webdisc.corpus.jsonl", "w") as w1, open("./data/webdisc/corpus/webdisc.corpus-tevatron.jsonl", "w") as w2:
        for docid in tqdm(webpages.keys()):

            count += 1
            doc = webpages[docid].replace("\t", " ").replace("\n", " ").replace("\r", " ")

            w1.write(json.dumps({"id": docid, "contents": doc}) + "\n")
            w2.write(json.dumps({"docid": docid, "title":"", "text": doc}) + "\n")

    logging.info(f"# doc: {count}")

    logging.info("Preprocessing conversations...")

    with open("./data/webdisc/raw/relevance_scores.txt", 'r') as r:
        qrel_raw = pytrec_eval.parse_qrel(r)


    mapping = {"full": "his-cur", "removelast": "his", "onlylast": "cur"}

    for q in ["full","removelast","onlylast"]:

        for s in ["test", "val", "train"]:


            qrels=defaultdict(dict)

            queries_doc = {}

            queries = load_queries(f"./data/webdisc/raw/queries_{q}/queries_{s}.tsv")

            num_turn = 0
            links = []

            #count=0

            for qid in queries.keys():
                num_turn += 1

                assert len(qrel_raw[qid]) == 1  # each query only has one relevant document

                docid = list(qrel_raw[qid].keys())[0]

                links.append(docid)

                queries_doc[qid] = limit_str_tokens(webpages[docid], 128)

                qrels[qid][docid]=1


            q_ = mapping[q]

            logging.info(f"Query {q_}, Set {s} has {num_turn} turns")
            logging.info(f"Query {q_}, Set {s} has {len(links)} webpage links")
            logging.info(f"Query {q_}, Set {s} has distinct {len(set(links))} webpage links")

            if q =="full":
                with open(f"./data/webdisc/qrels/webdisc.{s}.qrels.txt", "w") as w1:
                    for qid, doc2rel in qrels.items():
                        for doc, rel in doc2rel.items():
                            w1.write(f"{qid} Q0 {doc} {rel}\n")

                with open(f"./data/webdisc/queries/webdisc.{s}.queries.webpage.tsv", "w") as w1, open(
                        f"./data/webdisc/queries/webdisc.{s}.queries.webpage.jsonl", "w") as w2:
                    for qid, text in queries_doc.items():
                        w1.write(f"{qid}\t{text}\n")
                        w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")


            with open(f"./data/webdisc/queries/webdisc.{s}.queries.{q_}.tsv", "w") as w1, open(
                    f"./data/webdisc/queries/webdisc.{s}.queries.{q_}.jsonl", "w") as w2:
                for qid, text in queries.items():
                    w1.write(f"{qid}\t{text}\n")
                    w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")



if __name__ == '__main__':
    preprocess_webdisc()