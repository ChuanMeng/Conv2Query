import argparse
import os
import pytrec_eval
import json
from npdcg import calculate_npdcg

mapping = {"ndcg_cut_3": "ndcg@3",
           "ndcg_cut_5": "ndcg@5",
           "ndcg_cut_10": "ndcg@10",
           "ndcg_cut_20": "ndcg@20",
           "ndcg_cut_100": "ndcg@100",
           "ndcg_cut_1000": "ndcg@1000",
           "mrr_5": "mrr@5",
           "mrr_10": "mrr@10",
           "mrr_20": "mrr@20",
           "mrr_100": "mrr@100",
           "map_cut_10": "map@10",
           "map_cut_100": "map@100",
           "map_cut_1000": "map@1000",
           "recall_5": "recall@5",
           "recall_20": "recall@20",
           "recall_100": "recall@100",
           "recall_1000": 'recall@1000',
           "P_1": "precision@1",
           "P_3": "precision@3",
           "P_5": "precision@5",
           "P_10": "precision@10",
           "P_100": "precision@100",
           }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--qrels_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--rel_scale', type=int, required=True)

    args = parser.parse_args()


    with open(args.run_dir, 'r') as r:
        run = pytrec_eval.parse_run(r)

    with open(args.qrels_dir, 'r') as r:
        qrel = pytrec_eval.parse_qrel(r)

    print("len(list(run))", len(list(run)))
    print("len(list(qrel))", len(list(qrel)))


    # metric averaged on queries

    avg = {}

    # judged@10
    q2judge_10 = {}
    q2judge_20 = {}
    q2judge_100 = {}

    for qid, did_score in run.items():

        if qid not in qrel:
            continue

        sorted_did = [did for did, score in sorted(did_score.items(), key=lambda item: item[1], reverse=True)]
        judge_list = []

        for docid in sorted_did:
            if docid in qrel[qid]:
                judge_list.append(1)
            else:
                judge_list.append(0)

        q2judge_10[qid]=sum(judge_list[0:10])/10
        q2judge_20[qid]=sum(judge_list[0:20])/20
        q2judge_100[qid]=sum(judge_list[0:100])/100

    print('{}: {:.4f}'.format("judge_10", sum(q2judge_10.values())/len(q2judge_10)))
    print('{}: {:.4f}'.format("judge_20", sum(q2judge_20.values()) /len(q2judge_20)))
    print('{}: {:.4f}'.format("judge_100", sum(q2judge_100.values()) /len(q2judge_100)))

    avg[f"judge@{10}"] = sum(q2judge_10.values())/len(q2judge_10)
    avg[f"judge@{20}"] = sum(q2judge_20.values()) /len(q2judge_20)
    avg[f"judge@{100}"] = sum(q2judge_100.values()) /len(q2judge_100)

    run_5 = {}
    run_10 = {}
    run_20 = {}
    run_100 = {}

    for qid, did_score in run.items():
        sorted_did_score = [(did, score) for did, score in
                            sorted(did_score.items(), key=lambda item: item[1], reverse=True)]
        run_5[qid] = dict(sorted_did_score[0:5])
        run_10[qid] = dict(sorted_did_score[0:10])
        run_20[qid] = dict(sorted_did_score[0:20])
        run_100[qid] = dict(sorted_did_score[0:100])

    evaluator_ndcg = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20', 'ndcg_cut_100',
                                                           'ndcg_cut_1000'})
    results_ndcg = evaluator_ndcg.evaluate(run)

    results = {}
    for qid, _ in results_ndcg.items():
        results[qid] = {}
        for measure, score in results_ndcg[qid].items():
            results[qid][mapping[measure]] = score

    for q_id, pid_rel in qrel.items():
        for p_id, rel in pid_rel.items():
            if int(rel) >= args.rel_scale:
                qrel[q_id][p_id] = 1
            else:
                qrel[q_id][p_id] = 0

    evaluator_general = pytrec_eval.RelevanceEvaluator(qrel, {'map_cut_10', 'map_cut_100', 'map_cut_1000', 'recall_5', 'recall_20',
                                                              'recall_100', 'recall_1000', 'P_1', 'P_3', 'P_5', 'P_10',
                                                              "P_100"})
    results_general = evaluator_general.evaluate(run)

    for qid, _ in results.items():
        for measure, score in results_general[qid].items():
            results[qid][mapping[measure]] = score

    evaluator_rr = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})
    results_rr_5 = evaluator_rr.evaluate(run_5)
    results_rr_10 = evaluator_rr.evaluate(run_10)
    results_rr_20 = evaluator_rr.evaluate(run_20)
    results_rr_100 = evaluator_rr.evaluate(run_100)

    for qid, _ in results.items():
        results[qid][mapping["mrr_5"]] = results_rr_5[qid]['recip_rank']
        results[qid][mapping["mrr_10"]] = results_rr_10[qid]['recip_rank']
        results[qid][mapping["mrr_20"]] = results_rr_20[qid]['recip_rank']
        results[qid][mapping["mrr_100"]] = results_rr_100[qid]['recip_rank']

    for measure in mapping.values():
        overall = pytrec_eval.compute_aggregated_measure(measure, [result[measure] for result in results.values()])
        print('{}: {:.4f}'.format(measure, overall))
        avg[measure]=overall


    if "procis.test" in args.qrels_dir:
        # npdcg
        npdcg = []
        cuttoffs = [5, 10, 20]

        conv = {}
        for qid in qrel.keys():
            conv_id = qid.split("_")[0]
            if conv_id not in conv:
                conv[conv_id] = []
            conv[conv_id].append(qid)

        conv_max_turn = {}
        for conv_id in conv.keys():
            conv_max_turn[conv_id] =[]
            for qid in conv[conv_id]:
                conv_max_turn[conv_id].append(int(qid.split("_")[1]))
        for conv_id in conv_max_turn.keys():
            conv_max_turn[conv_id]=max(conv_max_turn[conv_id])

        # queries_proactive [[q11,q12,...],[q21,q22,...]...]
        for conv_id in conv.keys():
            retrieved = []

            for rank in range(1,(conv_max_turn[conv_id]+1)):
            #for qid in conv[conv_id]:
                qid = f"{conv_id}_{rank}"

                if qid in conv[conv_id]:
                    # per turn
                    retrieved_docs = [did for did, score in sorted(run[qid].items(), key=lambda item: item[1], reverse=True)]
                    correct_docs = [(did, socre) for did, socre in qrel[qid].items()]
                else:
                    retrieved_docs = []
                    correct_docs = []

                retrieved.append({'retrieved_docs': retrieved_docs, 'correct_docs': correct_docs})

            npdcg.append(calculate_npdcg(retrieved, [5, 10, 20]))

        # calculate average npdcg per cutoff, calculate_npdcg returns dict
        npdcg_avg = {c: sum([npdcg[i][c] for i in range(len(npdcg))]) / len(npdcg) for c in cuttoffs}
        for c in npdcg_avg.keys():
            print('npdcg@{}: {:.4f}'.format(c, npdcg_avg[c]))
            avg[f"npdcg@{c}"]=npdcg_avg[c]

    if args.output_path is not None:
        args.output_dir = "/".join(args.output_path.split("/")[0:-1])
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open(f"{args.output_path}", 'w') as w:
            w.write(json.dumps(avg))

