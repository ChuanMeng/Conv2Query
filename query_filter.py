import json
from argparse import ArgumentParser

import pytrec_eval
import collections

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--q2d_rerank_dir', type=str)
    parser.add_argument('--q2c_rerank_dir', type=str)
    parser.add_argument('--query_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--qrels_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="q2d") # "q2d", "q2c", "q2d_q2c"

    args = parser.parse_args()

    count = 0

    query = {}
    for i in range(args.num_chunks):
        print(f"Reading {args.query_dir}.chunk{i}.jsonl")
        with open(f"{args.query_dir}.chunk{i}.jsonl", 'r') as r:
            for line in r:
                data = json.loads(line.strip())
                docid_num = data["docid"]
                qtext = data["query"]
                query[docid_num] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

    print(f"# queries {len(query)}")

    if args.mode == "q2d":

        docid2docid_num2score = {}

        for i in range(args.num_chunks):
            print(f"Reading {args.q2d_rerank_dir}.chunk{i}.txt")
            with open(f"{args.q2d_rerank_dir}.chunk{i}.txt", 'r') as r:
                for line in r.readlines():

                    docid_num, docid, score = line.split('\t')
                    docid, num = docid_num.split("@")

                    if docid not in docid2docid_num2score:
                        docid2docid_num2score[docid] = {}

                    docid2docid_num2score[docid][docid_num] = float(score)

                    count += 1

        print(f"unique doc {len(docid2docid_num2score)}")

        docid2optimal_docid_num = {}
        doc2q_candi = []
        for docid in docid2docid_num2score:
            # print(doc2qs[docid])
            # print(len(doc2qs[docid]))
            doc2q_candi.append(len(docid2docid_num2score[docid]))
            docid2optimal_docid_num[docid] = sorted(docid2docid_num2score[docid].items(), key=lambda item: item[1], reverse=True)[0][0]

        print(sum(doc2q_candi) / len(doc2q_candi))

        # aggregate
        qid2qlist = collections.defaultdict(list)

        with open(args.qrels_dir, 'r') as r:
            qrels = pytrec_eval.parse_qrel(r)

        for qid, docid2rel in qrels.items():
            for docid, rel in docid2rel.items():
                if int(rel) >= 1:
                    qid2qlist[qid].append(query[docid2optimal_docid_num[docid]])

        print(len(qid2qlist))
        count = 0

        with open(f"{args.output_dir}-raw.tsv", "w") as w1, open(f"{args.output_dir}-raw.jsonl", 'w') as w2:
            for qid, docid2rel in qrels.items():
                for docid, rel in docid2rel.items():
                    if int(rel) >= 1:
                        count += 1

                        opimal_docid_num = docid2optimal_docid_num[docid]
                        qid_opimal_docid_num = f"{qid}@{docid2optimal_docid_num[docid]}"

                        w1.write(f"{qid_opimal_docid_num}\t{query[docid2optimal_docid_num[docid]]}\n")
                        w2.write(json.dumps({"query_id": qid_opimal_docid_num, "query": query[docid2optimal_docid_num[docid]]}) + "\n")

        with open(f"{args.output_dir}-concat.tsv", "w") as w1, open(f"{args.output_dir}-concat.jsonl", 'w') as w2:
            for qid, qlist in qid2qlist.items():
                qtext = " | ".join(qlist)
                w1.write(f"{qid}\t{qtext}\n")
                w2.write(json.dumps({"query_id": qid, "query": qtext}) + "\n")

        with open(f"{args.output_dir}-first.tsv", "w") as w1, open(f"{args.output_dir}-first.jsonl", 'w') as w2:
            for qid, qlist in qid2qlist.items():
                qtext = qlist[0]
                w1.write(f"{qid}\t{qtext}\n")
                w2.write(json.dumps({"query_id": qid, "query": qtext}) + "\n")

    elif args.mode == "q2c":

        qid_docid2docid_num2score = collections.defaultdict(dict)

        for i in range(args.num_chunks):
            print(f"Reading {args.q2c_rerank_dir}.chunk{i}.txt")
            with open(f"{args.q2c_rerank_dir}.chunk{i}.txt", 'r') as r:
                for line in r.readlines():

                    qid_docid_num, qid_, score = line.split('\t')
                    qid, docid, num = qid_docid_num.split("@")
                    docid_num =f"{docid}@{num}"
                    qid_docid = f"{qid}@{docid}"

                    assert qid_  == qid

                    qid_docid2docid_num2score[qid_docid][docid_num]=float(score)


        qid_docid2opimal_docid_num = {}
        for qid_docid in qid_docid2docid_num2score.keys():

            qid_docid2opimal_docid_num[docid] = sorted(qid_docid2docid_num2score[qid_docid].items(), key=lambda item: item[1], reverse=True)[0][0]

        qid2qlist = collections.defaultdict(list)

        with open(args.qrels_dir, 'r') as r:
            qrels = pytrec_eval.parse_qrel(r)

        for qid, docid2rel in qrels.items():
            for docid, rel in docid2rel.items():
                if int(rel) >= 1:
                    qid_docid =f"{qid}_{docid}"
                    if qid_docid not in qid_docid2opimal_docid_num:
                        continue
                    opimal_docid_num = qid_docid2opimal_docid_num[qid_docid]
                    qid2qlist[qid].append(query[opimal_docid_num])


        with open(f"{args.output_dir}-raw.tsv", "w") as w1, open(f"{args.output_dir}-raw.jsonl", 'w') as w2:
            for qid, docid2rel in qrels.items():
                for docid, rel in docid2rel.items():
                    if int(rel) >= 1:
                        qid_docid =f"{qid}_{docid}"

                        if qid_docid not in qid_docid2opimal_docid_num:
                            continue

                        opimal_docid_num = qid_docid2opimal_docid_num[qid_docid]
                        qid_opimal_docid_num = f"{qid}@{opimal_docid_num}"

                        w1.write(f"{qid_opimal_docid_num}\t{query[opimal_docid_num]}\n")
                        w2.write(json.dumps({"query_id": qid_opimal_docid_num, "query": query[opimal_docid_num]}) + "\n")

        with open(f"{args.output_dir}-concat.tsv", "w") as w1, open(f"{args.output_dir}-concat.jsonl", 'w') as w2:
            for qid, qlist in qid2qlist.items():
                qtext = " | ".join(qlist)
                w1.write(f"{qid}\t{qtext}\n")
                w2.write(json.dumps({"query_id": qid, "query": qtext}) + "\n")

        with open(f"{args.output_dir}-first.tsv", "w") as w1, open(f"{args.output_dir}-first.jsonl", 'w') as w2:
            for qid, qlist in qid2qlist.items():
                qtext = qlist[0]
                w1.write(f"{qid}\t{qtext}\n")
                w2.write(json.dumps({"query_id": qid, "query": qtext}) + "\n")

    elif args.mode=="q2d_q2c":

        docid2docid_num2score_q2d = {}

        for i in range(args.num_chunks):
            print(f"Reading {args.q2d_rerank_dir}.chunk{i}.txt")
            with open(f"{args.q2d_rerank_dir}.chunk{i}.txt", 'r') as r:
                for line in r.readlines():

                    docid_num, docid, score = line.split('\t')
                    docid, num = docid_num.split("@")

                    if docid not in docid2docid_num2score_q2d:
                        docid2docid_num2score_q2d[docid] = {}

                    docid2docid_num2score_q2d[docid][docid_num] = float(score)
                    count += 1

        qid_docid2docid_num2score_q2c = collections.defaultdict(dict)

        for i in range(args.num_chunks):
            print(f"Reading {args.q2c_rerank_dir}.chunk{i}.txt")
            with open(f"{args.q2c_rerank_dir}.chunk{i}.txt", 'r') as r:
                for line in r.readlines():
                    qid_docid_num, qid_, score = line.split('\t')
                    qid, docid, num = qid_docid_num.split("@")
                    docid_num = f"{docid}@{num}"
                    qid_docid = f"{qid}@{docid}"

                    assert qid_ == qid

                    qid_docid2docid_num2score_q2c[qid_docid][docid_num] = float(score)

                    count += 1

        # aggregate q2d and q2c
        qid_docid2docid_num2score = collections.defaultdict(dict)

        with open(args.qrels_dir, 'r') as r:
            qrels = pytrec_eval.parse_qrel(r)

        for qid, docid2rel in qrels.items():
            for docid, rel in docid2rel.items():
                if int(rel) >= 1:
                    qid_docid =f"{qid}_{docid}"

                    for docid_num in docid2docid_num2score_q2d[docid].keys():
                        score_q2d = docid2docid_num2score_q2d[docid][docid_num]
                        score_q2c = 0
                        if qid_docid in qid_docid2docid_num2score_q2c:
                            score_q2c = qid_docid2docid_num2score_q2c[qid_docid][docid_num]

                        score = score_q2d+score_q2c
                        qid_docid2docid_num2score[qid_docid][docid_num]=score

        qid_docid2opimal_docid_num = {}
        for qid_docid in qid_docid2docid_num2score.keys():
            qid_docid2opimal_docid_num[docid] = sorted(qid_docid2docid_num2score[qid_docid].items(), key=lambda item: item[1], reverse=True)[0][0]

        qid2qlist = collections.defaultdict(list)

        for qid, docid2rel in qrels.items():
            for docid, rel in docid2rel.items():
                if int(rel) >= 1:
                    qid_docid =f"{qid}_{docid}"
                    if qid_docid not in qid_docid2opimal_docid_num:
                        raise Exception
                    opimal_docid_num = qid_docid2opimal_docid_num[qid_docid]
                    qid2qlist[qid].append(query[opimal_docid_num])

        with open(f"{args.output_dir}-raw.tsv", "w") as w1, open(f"{args.output_dir}-raw.jsonl", 'w') as w2:
            for qid, docid2rel in qrels.items():
                for docid, rel in docid2rel.items():
                    if int(rel) >= 1:
                        qid_docid =f"{qid}_{docid}"

                        if qid_docid not in qid_docid2opimal_docid_num:
                            continue

                        opimal_docid_num = qid_docid2opimal_docid_num[qid_docid]
                        qid_opimal_docid_num = f"{qid}@{opimal_docid_num}"

                        w1.write(f"{qid_opimal_docid_num}\t{query[opimal_docid_num]}\n")
                        w2.write(json.dumps({"query_id": qid_opimal_docid_num, "query": query[opimal_docid_num]}) + "\n")

        with open(f"{args.output_dir}-concat.tsv", "w") as w1, open(f"{args.output_dir}-concat.jsonl", 'w') as w2:
            for qid, qlist in qid2qlist.items():
                qtext = " | ".join(qlist)
                w1.write(f"{qid}\t{qtext}\n")
                w2.write(json.dumps({"query_id": qid, "query": qtext}) + "\n")

        with open(f"{args.output_dir}-first.tsv", "w") as w1, open(f"{args.output_dir}-first.jsonl", 'w') as w2:
            for qid, qlist in qid2qlist.items():
                qtext = qlist[0]
                w1.write(f"{qid}\t{qtext}\n")
                w2.write(json.dumps({"query_id": qid, "query": qtext}) + "\n")

    else:
        Exception


