import sys
sys.path.append('./')
import json
from tqdm import tqdm
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def token_num(text):
    tokens = text.split(" ")
    return len(tokens)

def remove_newlines_tabs(text):
    return text.replace("\n", " ").replace("\t", " ").replace("\r", "")

def truncate_left_tokens(text, limit):
    tokens = text.split(" ")
    return " ".join(tokens[-limit:])

def limit_str_tokens(text, limit):
    tokens = text.split(" ")
    return " ".join(tokens[:limit])

def limit_turns_tokens(texts, limit):
    # texts: [u1, u2,...]
    added_tokens = 0
    trunc_texts = []
    # count from the end
    for text in reversed(texts):
        trunc_text_tokens = []
        tokens = text.split(" ")
        for token in tokens:
            if added_tokens == limit:
                break
            trunc_text_tokens.append(token)
            added_tokens += 1
        trunc_texts.append(" ".join(trunc_text_tokens))

    trunc_texts = [t for t in trunc_texts if t != ""]
    trunc_texts = reversed(trunc_texts)
    return " | ".join(trunc_texts)

def prepare_query(d, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100):
    # dict_keys(['post', 'thread', 'wiki_links', 'annotations'])

    last_k_turns = 0
    use_title = True
    use_content = True

    turns = [t["text"] for t in d["thread"]]

    # utterances
    if turns_max_tokens > 0:
        # [-0:] return the full list
        query_turns = limit_turns_tokens(turns[-last_k_turns:], turns_max_tokens)
    else:
        #query_turns = " ".join(turns[-last_k_turns:]) # Connect with spaces？？？
        query_turns = " | ".join(turns[-last_k_turns:])

    # title
    query_title = d["post"]["title"] if use_title else ""
    if title_max_tokens > 0:
        query_title = limit_str_tokens(query_title, title_max_tokens)

    # firs utterance
    query_content = d["post"]["text"] if use_content else ""
    if post_max_tokens > 0:
        query_content = limit_str_tokens(query_content, post_max_tokens)

    # Consider both the title and the initial post as individual utterances.
    query = remove_newlines_tabs(" | ".join([query_title, query_content, query_turns]))

    return query


def preprocess_procis():
    # collection

    logging.info("Preprocessing corpus...")

    docid2text={}
    title2docid = {}
    count=0

    with open(
            "data/procis/raw/aa/collection.jsonl") as r, open("data/procis/corpus/procis.corpus.jsonl/procis.corpus.jsonl", "w") as w1, open("data/procis/corpus/procis.corpus-tevatron.jsonl", "w") as w2:

        for line in tqdm(r):
            d = json.loads(line)
            count += 1
            doc = d["wiki"].replace('_', ' ') + ': ' + d["contents"]
            doc = doc.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            docid = f"procis_{count}"

            title = d["wiki"].replace('_', ' ')
            text =  d["contents"].replace("\t", " ").replace("\n", " ").replace("\r", " ")

            docid2text[docid] = doc

            if d["wiki"] not in title2docid:
                title2docid[d["wiki"]]=docid
            else:
                logging.info("Repetitive title: {}\ndoc for the repetitive title:\n{}\n{}\n\n{}\n{}".format(d["wiki"],title2docid[d["wiki"]],docid2text[title2docid[d["wiki"]]],docid,doc))

            w1.write(json.dumps({"id": docid, "contents": doc}) + "\n")
            w2.write(json.dumps({"docid": docid, "title": title, "text": text}) + "\n")


    logging.info(f"# doc: {count}")
    logging.info(f"# title: {len(title2docid)}")

    logging.info("Preprocessing conversations...")

    thre ={"train-filtered100":100,"train-filtered1000":1000, "train-filtered1500": 1500}

    for s in ["dev", "future_dev", "test", "train-filtered1000","train-filtered100", "train-filtered1500"]:

        id_conv2score = {}
        queries_conv = {}  # queries, one per conversation
        queries_his = {}
        queries_his_cur = {}
        queries_cur = {}
        queries_title_link = {}
        queries_title_manual = {}


        qrels_conv_link = {}
        qrels_conv_manual = {}

        qrels_turn_link = {}
        qrels_turn_manual = {}

        id_conv = 0

        num_turn = 0

        num_turn_w_link = 0
        num_turn_w_anno = 0

        num_link_conv = 0
        num_link_turn = 0

        num_anno_conv = 0
        num_anno_turn = 0

        num_link_conv_rep = 0
        num_link_turn_rep = 0

        num_anno_conv_rep = 0
        num_anno_turn_rep = 0


        if s in ["train-filtered1000","train-filtered100", "train-filtered1500"]:
            s_= "train"
        else:
            s_= s

        with open(f'./data/procis/raw/{s_}.jsonl') as r:
            for line in tqdm(r):
                d = json.loads(line)

                id_conv += 1

                qid_conv = str(id_conv)

                if s in ["train-filtered1000","train-filtered100",  "train-filtered1500"]:
                    if d['post']['score']<thre[s]:
                        # skip the current conv
                        continue

                if qid_conv in ["15266","1768414","1768948","2072677","2497208","2651077","991025"]: # noisy datapoint
                    continue

                id_conv2score[qid_conv] = d['post']['score']

                query_conv = prepare_query(d, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)

                queries_conv[qid_conv] = query_conv

                qrels_conv_link[qid_conv] = {}
                qrels_conv_manual[qid_conv] = {}


                for title in d['wiki_links']:
                    num_link_conv_rep+=1
                    if title2docid[title] not in qrels_conv_link[qid_conv]:
                        qrels_conv_link[qid_conv][title2docid[title]] = 1

                        num_link_conv += 1
                    else:
                        raise Exception

                if s == "test":
                    for annotation in d['annotations']:
                        num_anno_conv_rep += 1

                        if title2docid[annotation['wiki']] not in qrels_conv_manual[qid_conv]:
                            qrels_conv_manual[qid_conv][title2docid[annotation['wiki']]] = annotation['score']

                            num_anno_conv+=1
                        else:
                            # should not contain repetitive relevance judgments
                            assert qrels_conv_manual[qid_conv][title2docid[annotation['wiki']]] == annotation['score']

                id_turn = 0

                for i in range(len(d['thread'])):


                    num_turn+=1
                    id_turn += 1
                    qid_turn = f"{id_conv}_{id_turn}"

                    if s in ["train-filtered100", "train-filtered1000", "train-filtered1500", "train","dev", "future_dev"]:
                        if len(d['thread'][i]['wiki_links']) == 0:
                            continue

                    if s == "test":
                        if len(d['thread'][i]['annotations'])==0:
                            continue


                    d_his = d.copy()
                    d_his_cur = d.copy()

                    d_his['thread'] = d_his['thread'][:i]  # 0,1,2,...
                    d_his_cur['thread'] = d_his_cur['thread'][:i + 1]  # 1,2,3...

                    query_his = prepare_query(d_his, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100).strip()  # the first query is post's title and text, no thread
                    query_his_cur = prepare_query(d_his_cur, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100).strip()
                    query_cur = limit_str_tokens(remove_newlines_tabs(d['thread'][i]["text"]),300).strip() # truncate from the right side

                    if query_his=="" or query_his is None:
                        query_his = "[link]"

                    if query_his_cur=="" or query_his_cur is None:
                        query_his_cur = "[link]"

                    if query_cur =="" or query_cur is None:
                        query_cur = "[link]"


                    queries_his[qid_turn] = query_his
                    queries_his_cur[qid_turn] = query_his_cur
                    queries_cur[qid_turn] = query_cur


                    titles_link = []
                    titles_manual = []

                    qrels_turn_link[qid_turn] = {}
                    qrels_turn_manual[qid_turn] = {}

                    if len(d['thread'][i]['wiki_links']) > 0:
                        num_turn_w_link += 1

                    for title in d['thread'][i]['wiki_links']:
                        num_link_turn_rep+=1

                        if title2docid[title] not in qrels_turn_link[qid_turn]:
                            qrels_turn_link[qid_turn][title2docid[title]] = 1

                            titles_link.append(title)

                            num_link_turn += 1
                        else:
                            pass

                    concat_title_link = " ".join([title.replace('_', ' ') for title in titles_link])
                    if concat_title_link != "":
                        queries_title_link[qid_turn] = concat_title_link


                    if s == "test":
                        if len(d['thread'][i]['annotations'])>0:
                            num_turn_w_anno+=1

                        for annotation in d['thread'][i]['annotations']:
                            num_anno_turn_rep+=1

                            if title2docid[annotation['wiki']] not in qrels_turn_manual[qid_turn]:
                                qrels_turn_manual[qid_turn][title2docid[annotation['wiki']]] = annotation['score']

                                num_anno_turn+=1

                                titles_manual.append(annotation['wiki'])
                            else:
                                #should not contain repetitive relevance judgments
                                assert qrels_turn_manual[qid_turn][title2docid[annotation['wiki']]] == annotation['score']
                                #logging.info("Found repetitive wiki links:\n{}\n{}".format(d['wiki_links'],d['thread'][i]['wiki_links']))

                        concat_title_manual = " ".join([title.replace('_', ' ') for title in titles_manual])
                        if concat_title_manual != "":
                            queries_title_manual[qid_turn] = concat_title_manual


        logging.info(f"Set {s} has {id_conv} conversations")
        logging.info(f"Set {s} has {num_turn} turns")
        logging.info(f"Average number of turns per conversation: {num_turn/id_conv}")

        logging.info(f"Set {s} has {num_turn_w_link} turns w/ wiki links")

        logging.info(f"Set {s} contains {num_link_conv_rep} conversation-level links")
        logging.info(f"Set {s} contains {num_link_conv} conversation-level links (deduplication)")
        logging.info(f"Average number of conversation-level links per conversation: {num_link_conv / id_conv}")
        logging.info(f"Average number of conversation-level links per query: {num_link_conv / num_turn_w_link}")

        logging.info(f"Set {s} contains {num_link_turn_rep} turn-level links")
        logging.info(f"Set {s} contains {num_link_turn} turn-level links (deduplication)")
        logging.info(f"Average number of turn-level links per conversation: {num_link_turn / id_conv}")
        logging.info(f"Average number of turn-level links per query: {num_link_turn / num_turn_w_link}")


        if s == "test":
            logging.info(f"Set {s} has {num_turn_w_anno} turns w/ annotations")

            logging.info(f"Set {s} contains {num_anno_conv_rep} conversation-level annotations")
            logging.info(f"Set {s} contains {num_anno_conv} conversation-level annotations (deduplication)")
            logging.info(
                f"Average number of conversation-level annotations per conversation: {num_anno_conv / id_conv}")
            logging.info(
                f"Average number of conversation-level annotations per query: {num_anno_conv / num_turn_w_anno}")


            logging.info(f"Set {s} contains {num_anno_turn_rep} turn-level annotations")
            logging.info(f"Set {s} contains {num_anno_turn} turn-level annotations (deduplication)")
            logging.info(f"Average number of turn-level annotations per conversation: {num_anno_turn/id_conv}")
            logging.info(f"Average number of turn-level annotations per query: {num_anno_turn/num_turn_w_anno}")


        with open(f"./data/procis/queries/procis.{s}.queries.conv.tsv", "w") as w1,open(f"./data/procis/queries/procis.{s}.queries.conv.jsonl", "w") as w2:
            for qid, text in queries_conv.items():
                w1.write(f"{qid}\t{text}\n")
                w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")

        with open(f"./data/procis/queries/procis.{s}.queries.his.tsv", "w") as w1,open(f"./data/procis/queries/procis.{s}.queries.his.jsonl", "w") as w2:
            for qid, text in queries_his.items():
                w1.write(f"{qid}\t{text}\n")
                w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")

        with open(f"./data/procis/queries/procis.{s}.queries.his-cur.tsv", "w") as w1,open(f"./data/procis/queries/procis.{s}.queries.his-cur.jsonl", "w") as w2:
            for qid, text in queries_his_cur.items():
                w1.write(f"{qid}\t{text}\n")
                w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")

        with open(f"./data/procis/queries/procis.{s}.queries.cur.tsv", "w") as w1,open(f"./data/procis/queries/procis.{s}.queries.cur.jsonl", "w") as w2:
            for qid, text in queries_cur.items():
                w1.write(f"{qid}\t{text}\n")
                w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")

        with open(f"./data/procis/queries/procis.{s}.queries.title-link.tsv", "w") as w1,open(f"./data/procis/queries/procis.{s}.queries.title-link.jsonl", "w") as w2:
            for qid, text in queries_title_link.items():
                w1.write(f"{qid}\t{text}\n")
                w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")


        with open(f"./data/procis/qrels/procis.{s}.qrels.conv-link.txt", "w") as w1:
            for qid, doc2rel in qrels_conv_link.items():
                for doc, rel in doc2rel.items():
                    w1.write(f"{qid} Q0 {doc} {rel}\n")

        with open(f"./data/procis/qrels/procis.{s}.qrels.turn-link.txt", "w") as w1:
            for qid, doc2rel in qrels_turn_link.items():
                for doc, rel in doc2rel.items():
                    w1.write(f"{qid} Q0 {doc} {rel}\n")

        if s == "test":
            with open(f"./data/procis/queries/procis.{s}.queries.title-manual.tsv", "w") as w1,open(f"./data/procis/queries/procis.{s}.queries.title-manual.jsonl", "w") as w2:
                for qid, text in queries_title_manual.items():
                    w1.write(f"{qid}\t{text}\n")
                    w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")

            with open(f"./data/procis/qrels/procis.{s}.qrels.conv-manual.txt", "w") as w1:
                for qid, doc2rel in qrels_conv_manual.items():
                    for doc, rel in doc2rel.items():
                        w1.write(f"{qid} Q0 {doc} {rel}\n")

            with open(f"./data/procis/qrels/procis.{s}.qrels.turn-manual.txt", "w") as w1:
                for qid, doc2rel in qrels_turn_manual.items():
                    for doc, rel in doc2rel.items():
                        w1.write(f"{qid} Q0 {doc} {rel}\n")



if __name__ == '__main__':
    preprocess_procis()