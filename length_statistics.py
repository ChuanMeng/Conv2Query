from transformers import LlamaTokenizer
import numpy as np
import json
import tqdm
import argparse

def token_num(text):
    tokens = text.split(" ")
    return len(tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default=None)
    parser.add_argument("--query_dirs", type=str, nargs='+')

    args = parser.parse_args()


    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")

    doc_len_llama = []
    doc_len_space = []
    docid2text = dict()

    with open(f"{args.corpus_dir}") as r:
        for line in tqdm.tqdm(r):
            d = json.loads(line)
            docid2text[d["id"]] = d["contents"]
            doc_len_llama.append(len(tokenizer.tokenize(d["contents"])))
            doc_len_space.append(token_num(d["contents"]))

    average_token_length_llama = np.mean(doc_len_llama)
    average_token_length_space = np.mean(doc_len_space)
    print(f"Average document token length. llama: {average_token_length_llama:.2f}, space: {average_token_length_space:.2f}")


    for query_dir in args.query_dirs:

        query = {}
        query_len_llama = []
        query_len_space = []

        with open(f"{query_dir}", 'r') as r:
            for line in r.readlines():
                qid, qtext = line.split('\t')
                query[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")
                query_len_llama.append(len(tokenizer.tokenize(query[qid])))
                query_len_space.append(token_num(query[qid]))

        average_token_length_llama = np.mean(query_len_llama)
        average_token_length_space = np.mean(query_len_space)

        print( f"{query_dir}\n query token length. llama: {average_token_length_llama:.2f}, space: {average_token_length_space}")




