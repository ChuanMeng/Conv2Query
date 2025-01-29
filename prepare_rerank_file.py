import json
from argparse import ArgumentParser
import tqdm
import collections
import pytrec_eval

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--query_dir', type=str, required=True)
    parser.add_argument('--corpus_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default="q2d")  # "q2c
    parser.add_argument("--qrels_dir", type=str, default=None)

    args = parser.parse_args()

    docid2doc = dict()
    with open(args.corpus_dir) as r:
        for line in tqdm.tqdm(r):
            d = json.loads(line)
            if "docid" in d and args.mode =="q2d":
                docid2doc[d["docid"]] = d
            elif "query_id" in d and args.mode =="q2c":
                docid2doc[d["query_id"]] = {"docid": d["query_id"], "title": "", "text": d["query"]}


    # load generated queries
    #queries = []
    docid2queries =collections.defaultdict(list)
    num_query =0

    if ".tsv" in args.query_dir:
        with open(args.query_dir, 'r') as r:
            for line in r.readlines():
                docid_num, qtext = line.split('\t')
                docid,num = docid_num.split("@")
                #queries.append((docid_num, qtext.strip()))
                docid2queries[docid].append((docid_num, qtext.strip()))
                num_query+=1

    elif ".jsonl" in args.query_dir:
        with open(args.query_dir, 'r') as r:
            for line in r:
                data = json.loads(line.strip())

                docid_num = data['docid']
                qtext = data["query"]
                docid, num = docid_num.split("@")
                #queries.append((docid_num, qtext.strip()))
                docid2queries[docid].append((docid_num, qtext.strip()))
                num_query += 1

    else:
        raise Exception

    if args.mode == "q2d":

        count_exp = 0
        with open(args.output_dir, 'w') as w:
            for docid in docid2queries.keys():

                for docid_num, qtext in docid2queries[docid]:
                    docid, num = docid_num.split("@")

                    psg_info = docid2doc[docid]
                    psg_info['score'] = 0
                    psg_info['query_id'] = docid_num
                    psg_info['query'] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

                    if count_exp in [0, 2, 99, 100, 101, 199]:
                        print(psg_info)

                    w.write(json.dumps(psg_info) + '\n')

                    count_exp += 1

            print(f"# queries: {num_query}")
            print(f"# examples: {count_exp}")

    elif args.mode == "q2c":

        count_exp = 0
        count_remov = 0
        with open(args.qrels_dir, 'r') as r:
            qrels = pytrec_eval.parse_qrel(r)

        with open(args.output_dir, 'w') as w:
            for qid, docid2rel in qrels.items():
                for docid, rel in docid2rel.items():
                    if int(rel) >= 1:
                        for docid_num, qtext in docid2queries[docid]:

                            psg_info = docid2doc[qid]
                            psg_info['score'] = 0
                            psg_info['query_id'] = f"{qid}@{docid_num}"
                            psg_info['query'] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

                            if psg_info["text"].strip() in ["[link]","NULL",""]:
                                count_remov += 1
                                continue

                            if count_exp in [0, 2, 99, 100, 101, 199]:
                                print(psg_info)

                            w.write(json.dumps(psg_info) + '\n')
                            count_exp+=1

        print(f"write in # {count_exp} examples")
        print(f"invalid examples #{count_remov}")

    else:
        Exception


