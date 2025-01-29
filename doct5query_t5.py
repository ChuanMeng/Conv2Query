import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tqdm
import json
import pytrec_eval
import argparse

def load_data(args):
    docid2doctext = dict()
    with open(args.corpus_dir) as r:
        for line in tqdm.tqdm(r):
            d = json.loads(line)
            docid2doctext[d["id"]] = d["contents"]

    with open(args.qrels_dir, 'r') as r:
        qrels = pytrec_eval.parse_qrel(r)

    docid_rel = []
    docid_rel_set = set()

    for qid, docid2rel in qrels.items():
        for docid, rel in docid2rel.items():
            if int(rel)>=1 and docid not in docid_rel_set:
                docid_rel.append(docid)
                docid_rel_set.add(docid)

    examples = []

    for docid in docid_rel:
        example = {}
        example["example_id"] = docid
        example["input"] = docid2doctext[docid]
        examples.append(example)

    print(len(examples))
    print("First example:\n", examples[0]["example_id"], "\n", examples[0]["input"][:128])
    print("Second example:\n", examples[1]["example_id"], "\n", examples[1]["input"][:128])
    print("Last example:\n", examples[-1]["example_id"], "\n", examples[-1]["input"][:128])


    if args.num_chunks >1 and args.local_rank!=-1:
        n = len(examples)
        #chunk_size = (n + 3) // 4
        chunk_size = (n + args.num_chunks - 1) // args.num_chunks
        #print(chunk_size)
        chunks = [examples[i:i + chunk_size] for i in range(0, n, chunk_size)]

        assert len(chunks)==args.num_chunks

        if args.local_rank >= args.num_chunks:
            raise ValueError(f"Requested chunk index {args.local_rank} out of range. "
                             f"Total chunks: {args.num_chunks}")

        selected_chunk = chunks[args.local_rank]

        print(f"\nReturning chunk with the index {args.local_rank} from {args.num_chunks} chunks with {len(selected_chunk)} examples.")

        print("First example in this chunk:\n", selected_chunk[0]["example_id"],"\n",selected_chunk[0]["input"][:128])
        print("Second example in this chunk:\n", selected_chunk[1]["example_id"],"\n",selected_chunk[1]["input"][:128])
        print("Last example in this chunk:\n", selected_chunk[-1]["example_id"],"\n",selected_chunk[-1]["input"][:128])

        return selected_chunk
    else:
        return examples


class T5:
    def __init__(self, args):
        self.args=args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1', truncation_side=self.args.truncation_side)
        self.model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')

        self.model = self.model.to(self.device)
        self.model.eval()

    def transform(self, examples):
        it = range(0, len(examples), self.args.batch_size)

        for start_idx in tqdm.tqdm(it):
            rng = slice(start_idx, start_idx + self.args.batch_size)
            # Prepare inputs
            inputs = [example['input'] for example in examples[rng]]

            # padding=True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
            # enc = self.tokenizer.(], padding=True, truncation=True, max_length=args.max_input_length, return_tensors='pt')
            enc = self.tokenizer.batch_encode_plus(inputs, return_tensors='pt', padding='longest', truncation=True, max_length=self.args.max_input_length)

            enc = {k: v.to(self.device) for k, v in enc.items()}


            with torch.no_grad():
                predictions = self.model.generate(  # Use model.module when using DataParallel
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    max_length=self.args.max_target_length,
                    do_sample=False if self.args.query_num == 1 else True,
                    # do_sample：default is False。
                    # top_p=0.95,
                    top_k=None if self.args.query_num == 1 else 10,
                    num_return_sequences=self.args.query_num
                )


            predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True,clean_up_tokenization_spaces=True)

            #assert len(predictions) % args.query_num * args.batch_size

            for idx, example in enumerate(examples[rng]):
                #example["prediction"] = " ".join([predictions[idx]]*3)
                #example["prediction"] ="@".join(predictions[idx*args.query_num:(idx+1)*args.query_num]).strip() #concatenate
                example["prediction"] = predictions[idx * self.args.query_num:(idx + 1) * self.args.query_num]

                #print(example["prediction"])

        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default=None)
    parser.add_argument("--qrels_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--query_num", type=int, default=1)
    parser.add_argument("--truncation_side", type=str, default='right')

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    args.dataset_class = args.qrels_dir.split("/")[-3].split(".")[0]
    args.dataset_name = args.qrels_dir.split("/")[-1].split(".")[1]

    print(args.dataset_class)
    print(args.dataset_name)

    args.setup = f"{args.dataset_class}.{args.dataset_name}.queries.doct5query-{args.query_num}"

    args.output_dir_ = f"{args.output_dir}/{args.setup}"

    examples = load_data(args)
    t5 = T5(args)
    t5.transform(examples)

    with open(f"{args.output_dir_}.chunk{args.local_rank}.tsv", "w") as w1, open(f"{args.output_dir_}.chunk{args.local_rank}.jsonl", "w") as w2:
        for idx, example in enumerate(examples):
            docid = example["example_id"]
            queries = example["prediction"]

            assert len(example["prediction"])==args.query_num

            for idx in range(args.query_num):
                docid_idx = f"{docid}@{idx}"
                text = queries[idx]

                w1.write(f"{docid_idx}\t{text}\n")
                w2.write(json.dumps({'docid': docid_idx, "query": text}) + "\n")
