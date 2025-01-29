import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import tqdm
import json
import pytrec_eval
import argparse
import re
from typing import List, Optional

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


class LLamaQueryGenerator:
    def __init__(self, llama_path: str, max_tokens, peft_path: Optional[str] = None):
        self.llama_path = llama_path
        self.max_tokens = max_tokens
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llama_path)
        self.tokenizer.pad_token_id = 0  # making it different from the eos token
        self.tokenizer.padding_side = 'left'

        self.model = LlamaForCausalLM.from_pretrained(
            self.llama_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.bfloat16,
                # bnb_4bit_use_double_quant=True,
            ),
            torch_dtype=torch.bfloat16,
            device_map=torch.device('cuda')
        )

        if peft_path is not None:
            self.peft_config = PeftConfig.from_pretrained(peft_path)
            self.model = PeftModel.from_pretrained(self.model, peft_path)

        self.model.eval()

    @torch.no_grad()
    def generate(self, documents: List[str], **kwargs):
        assert 'num_return_sequences' in kwargs
        n_ret_seq = kwargs['num_return_sequences']
        inputs = self.prompt_and_tokenize(documents)
        outputs = self.model.generate(**inputs, **kwargs)
        predicted_queries = []
        for d in self.tokenizer.batch_decode(outputs, skip_special_tokens=True):
            predicted_queries.append(re.sub(r"\s{2,}", ' ', d.rsplit('\n---\n', 1)[-1]))
        return [predicted_queries[i: i + n_ret_seq] for i in range(0, len(predicted_queries), n_ret_seq)]

    @torch.no_grad()
    def prompt_and_tokenize(self, documents: List[str]):
        prompts = [f'Predict possible search queries for the following document:\n{document}' for document in
                   documents]
        encoded = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True, max_length=self.max_tokens, truncation=True)

        for input_id in encoded['input_ids']:
            # Check if last three items are not [13, 5634, 13] i.e. \n---\n
            if not torch.equal(input_id[-3:], torch.tensor([13, 5634, 13])):
                # Replace them
                input_id[-3:] = torch.tensor([13, 5634, 13])

        encoded['input_ids'] = encoded['input_ids'].to(torch.device('cuda'))
        encoded['attention_mask'] = encoded['attention_mask'].to(torch.device('cuda'))
        return encoded

def generate_queries_and_save(args, query_generator, doc_batch, doc_ids):
    queries_list = query_generator.generate(
        doc_batch,
        num_return_sequences=args.query_num,
        max_new_tokens= args.max_new_tokens,
        do_sample = False if args.query_num == 1 else True,
        top_k=None if args.query_num == 1 else 50,
        top_p=None if args.query_num == 1 else 0.95,
    )

    with open(f"{args.output_dir_}.jsonl", 'a', encoding='utf-8') as out:
        for idx_doc, queries in enumerate(queries_list):
            docid=doc_ids[idx_doc]

            for idx_q in range(len(queries)):
                docid_num = f"{docid}@{idx_q}"
                json.dump({'docid': docid_num, 'query': queries[idx_q]}, out)
                out.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default=None)
    parser.add_argument("--qrels_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--query_num", type=int, default=1)

    parser.add_argument("--token", type=str)
    parser.add_argument("--cache_dir", type=str)

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    args.dataset_class = args.qrels_dir.split("/")[-3].split(".")[0]
    args.dataset_name = args.qrels_dir.split("/")[-1].split(".")[1]

    args.llama_path = "meta-llama/Llama-2-7b-hf"
    args.peft_path = "soyuj/llama2-doc2query"

    args.setup = f"{args.dataset_class}.{args.dataset_name}.queries.docllama2query-{args.query_num}"

    if args.local_rank!=-1 and args.num_chunks>1:
        args.output_dir_ = f"{args.output_dir}/{args.setup}.chunk{args.local_rank}"
    else:
        args.output_dir_ = f"{args.output_dir}/{args.setup}"

    generator = LLamaQueryGenerator(llama_path=args.llama_path, max_tokens=args.max_input_length, peft_path=args.peft_path)

    batch = []
    ids = []

    examples = load_data(args)

    for example in tqdm.tqdm(examples):
        batch.append(example["input"])
        ids.append(example["example_id"])

        if len(batch) == args.batch_size:
            generate_queries_and_save(args, generator, batch, ids)
            batch = []
            ids = []

    if batch:
        generate_queries_and_save(args, generator, batch, ids)
