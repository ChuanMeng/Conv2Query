import json
import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--query_dir', type=str, required=True)

    args = parser.parse_args()

    query = {}
    with open(args.query_dir, 'r') as r:
        for line in r.readlines():
            qid, qtext = line.split('\t')
            query[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

    cur_query_type = args.source_dir.split("/")[-1].split(".")[2].split("--neg20")[0]
    new_query_type = args.query_dir.split("/")[-1].split(".")[3]

    args.output_dir = args.source_dir.replace(cur_query_type, new_query_type,1)

    print(args.source_dir)
    print(args.output_dir)

    with open(args.source_dir, 'r') as r, open(args.output_dir, 'w') as w:
        for line in tqdm.tqdm(r):
            ex = json.loads(line)
            qid = ex["query_id"]
            ex["query"] = query[qid]

            w.write(json.dumps(ex) + "\n")