import json
import argparse
import tqdm
import pytrec_eval
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default=None)
    parser.add_argument("--query_dir", type=str, default=None)
    parser.add_argument("--qrels_dir", type=str, default=None)
    parser.add_argument("--run1_dir", type=str, default=None)
    parser.add_argument("--run2_dir", type=str, default=None)
    parser.add_argument("--run3_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--num_neg", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()


    args.dataset_class = args.qrels_dir.split("/")[-1].split(".")[0]
    args.dataset_name = args.qrels_dir.split("/")[-1].split(".")[1]
    args.query_type = args.query_dir.split("/")[-1].split(".")[3]

    args.ranker = [".".join(args.run1_dir.split("/")[-1].split(".")[3:-1])+f"-{args.k}"]

    if args.run2_dir is not None:
        args.ranker.append(".".join(args.run2_dir.split("/")[-1].split(".")[3:-1])+f"-{args.k}")

    if args.run3_dir is not None:
        args.ranker.append(".".join(args.run3_dir.split("/")[-1].split(".")[3:-1])+f"-{args.k}")

    args.ranker = "-".join(args.ranker)

    #args.setup = f"{args.dataset_class}.{args.dataset_name}.{args.query_type}--neg{args.num_neg}-{args.ranker}.jsonl

    args.setup = f"{args.dataset_class}.{args.dataset_name}.{args.query_type}--neg{args.num_neg}.jsonl"

    print(args.setup)

    args.output_dir_ = f"{args.output_dir}/{args.setup}"

    random.seed(args.random_seed)

    docid2doc = dict()
    with open(args.corpus_dir) as r:
        for line in tqdm.tqdm(r):
            d = json.loads(line)
            docid2doc[d["docid"]] = d

    query = {}
    with open(args.query_dir, 'r') as r:
        for line in r.readlines():
            qid, qtext = line.split('\t')
            query[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

    with open(args.qrels_dir, 'r') as r:
        qrels = pytrec_eval.parse_qrel(r)

    with open(args.run1_dir, 'r') as r:
        run1 = pytrec_eval.parse_run(r)

    runs = [run1]

    if args.run2_dir is not None:
        with open(args.run2_dir, 'r') as r:
            run2 = pytrec_eval.parse_run(r)
        runs.append(run2)

    if args.run3_dir is not None:
        with open(args.run3_dir, 'r') as r:
            run3 = pytrec_eval.parse_run(r)
        runs.append(run3)

    examples = []

    skip = 0

    for qid, docid2rel in tqdm.tqdm(qrels.items()):

        # one query
        example = {}
        example["query_id"] = qid
        example["query"] = query[qid]
        example["positive_passages"] = []
        example["negative_passages"] = []

        # add positive doc
        positive_docids = set()
        for docid, rel in docid2rel.items():
            if int(rel) >= 1: # for each positive doc

                positive_passage = {}
                positive_passage["docid"] = docid

                positive_docids.add(docid)

                positive_passage["title"] = docid2doc[docid]["title"]
                positive_passage["text"] = docid2doc[docid]["text"]

                example["positive_passages"].append(positive_passage)

        # sample negative doc
        negatives_docids = set()
        for run in runs:
            sorted_did = [did for did, score in sorted(run[qid].items(), key=lambda item: item[1], reverse=True)][:args.k]
            negatives_docids.update(set(sorted_did))

        negatives_docids = negatives_docids-positive_docids

        if len(negatives_docids)<args.num_neg:
            skip+=1
            continue

        negatives_docids_sample = random.sample(list(negatives_docids), args.num_neg)

        if len(examples)==0 or len(examples)==1000 or len(examples)==10000:
            print(negatives_docids_sample)

        for docid in negatives_docids_sample:

            negative_passage = {}
            negative_passage["docid"] = docid

            negative_passage["title"] = docid2doc[docid]["title"]
            negative_passage["text"] = docid2doc[docid]["text"]

            example["negative_passages"].append(negative_passage)


        examples.append(example)

    print(f"# examples: {len(examples)}")
    print(f"# skip {skip}")

    with open(args.output_dir_, "w") as w:
        for example in examples:
            w.write(json.dumps(example) + "\n")