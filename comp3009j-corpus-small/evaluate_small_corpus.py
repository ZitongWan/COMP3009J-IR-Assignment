"""
StudentID: 22207296
StudentName: Zitong Wan
"""

import os
import math
import time
import argparse
from collections import defaultdict


class Evaluator:
    def __init__(self, qrels_path, results_path):
        # rel: relevant   ret: retrieved
        self.rel = self.load_qrels(qrels_path)
        self.ret = self.load_results(results_path)

    def load_qrels(self, file_path):
        """Obtain Rel --- relevant set"""
        # relevant: {'queryId1': {docId1: relevanceScore1, docId2: relevanceScore2, ...}, 'queryId2: {docId1: relevanceScore1, docId2: relevanceScore2, ...}'}
        # This relevance is the dictionary corresponding to each query_id.
        # In the dictionary corresponding to each query_id, the key is doc_id and the value is relevance_score.
        # eg: {'1': {'184': 2, '29': 2, '31': 2, ...}, '2': {'12': 1, '15': 2, ...}, ...}
        relevant = defaultdict(dict)
        with open(file_path, 'r') as f:
            for line in f:
                query_id, _, doc_id, relevance_socre = line.strip().split()
                relevant[query_id][doc_id] = int(relevance_socre)

        return relevant

    def load_results(self, file_path):
        """Obtain Ret --- retrieved set"""
        # retrieved: {'queryId1: [{'doc_id': id1, 'rank': rank1, 'score': bm25_score1}, {'doc_id': id2, ...}, ...]'}
        # This retrieved is a list corresponding to each query_id, where each element in the list is a dictionary containing doc_id, rank, and score.
        # eg: {'1': [{'doc_id': '51', 'rank': 1, 'score': 39.2047}, {'doc_id': '486', 'rank': 2, 'score': 34.2867}, ...], '3': [{...}]}
        retrieved = defaultdict(list)
        with open(file_path, 'r') as f:
            for line in f:
                query_id, doc_id, rank, score = line.strip().split()
                retrieved[query_id].append({
                    'doc_id': doc_id,
                    'rank': int(rank),
                    'score': float(score)
                })

        return retrieved

    def get_rel_ret(self, rel, ret, query_id):
        """Obtain the corresponding RelRet based on QueryId --- rel_rets"""
        # rel_rets: {rank1: {'doc_id': id1, 'rank', rank1, 'score:' score1, 'relevance': relevance1}, rank2 : {...}, ...}
        # Here rel_rets is actually adding a relevance based on rel in ret, and then using the rank in ret as the key, the corresponding value is ret, such as:
        # --- rel_rets[rank] = ret(After add relevance) ---"""
        # eg: for query_id = xxx:
        # The target rel_rets: {1: {'doc_id': '51', 'rank': 1, 'score': 39.2047, 'relevance': 3}, 2: {'doc_id': '486', 'rank': 2, 'score': 34.2867, 'relevance': 1, ...}
        rel_docs = rel[query_id]
        ret_docs = ret[query_id]
        rel_rets = {}
        for ret_doc in ret_docs:
            rank = ret_doc['rank']
            ret_doc['relevance'] = rel_docs[ret_doc['doc_id']] if ret_doc['doc_id'] in rel_docs else -1  # If not relevant, set to -1
            rel_rets[rank] = ret_doc

        return rel_rets

    def precision(self, rel, ret):
        """Calculate the precision"""
        precisions = 0
        for query_id in ret:
            rel_rets = self.get_rel_ret(rel, ret, query_id)
            rel_ret_num = sum(1 for rel_doc in rel_rets.values() if rel_doc['relevance'] > 0)  # Obtain the number of relevant documents in rel_rets
            precisions += rel_ret_num / len(ret[query_id])

        return precisions / len(ret) if len(ret) > 0 else 0

    def recall(self, rel, ret):
        """Calculate the recall"""
        recalls = 0
        for query_id in ret:
            rel_rets = self.get_rel_ret(rel, ret, query_id)
            rel_ret_num = sum(1 for rel_doc in rel_rets.values() if rel_doc['relevance'] > 0)  # Obtain the number of relevant documents in rel_rets
            rel_num = sum(1 for rel_doc in rel[query_id].values() if rel_doc > 0)  # Obtain the quantity of all relevant documents in rel

            recalls += rel_ret_num / rel_num

        return recalls / len(ret) if len(ret) > 0 else 0

    def r_precision(self, rel, ret):
        """Calculate the R-precision"""
        r_precisions = 0
        for query_id in ret:
            rel_rets = self.get_rel_ret(rel, ret, query_id)
            rel_num = sum(1 for rel_doc in rel[query_id].values() if rel_doc > 0)  # Obtain the quantity of all relevant documents in rel
            top_num = 0  # Retrieve the relevance count of the first rel_num documents in rel_rets
            for i, rel_doc in enumerate(rel_rets.values(), start=1):
                if i <= rel_num and rel_doc['relevance'] > 0:
                    top_num += 1

            r_precisions += top_num / rel_num

        return r_precisions / len(ret) if len(ret) > 0 else 0

    def precision_15(self, rel, ret):
        """Calculate the precision@15"""
        precisions = 0
        for query_id in ret:
            # Get 15 documents in ret [query_id], if it is less than 15, all documents in ret [query_id] will be taken.
            tok_k_ret = ret[query_id][:15] if len(ret[query_id]) >= 15 else ret[query_id]
            rel_rets = self.get_rel_ret(rel, {query_id: tok_k_ret}, query_id)
            rel_ret_num = sum(1 for rel_doc in rel_rets.values() if rel_doc['relevance'] > 0)  # Obtain the number of relevant documents in rel_rets
            query_id_doc_num = len(tok_k_ret)
            precisions += rel_ret_num / query_id_doc_num

        return precisions / len(ret) if len(ret) > 0 else 0

    def ndcg_15(self, rel, ret, n=15):
        """Calculate the NDCG@15"""
        ndcgs = 0
        for query_id in ret:
            rel_rets = self.get_rel_ret(rel, ret, query_id)
            prev_dcg = 0  # Record the previous DCG value for replacement
            for rank, doc in rel_rets.items():
                # Add the 'doc' key and its corresponding value directly to rel_rets here
                if rank == 1:
                    doc['dcg'] = float(doc['relevance']) if doc['relevance'] > 0 else 0
                else:
                    doc['dcg'] = (doc['relevance'] / math.log2(rank) + prev_dcg) if doc['relevance'] > 0 else prev_dcg
                prev_dcg = doc['dcg']

            relevant_doc_scores = rel[query_id]
            sorted_docs = sorted(relevant_doc_scores.values(), reverse=True)[:len(rel_rets)]  # Sort in descending order by relevance score

            idcg_values = {}
            prev_idcg = 0  # Record the previous idcg value
            for rank, doc in enumerate(sorted_docs, start=1):
                if rank == 1:
                    current_idcg = float(doc) if doc > 0 else 0
                else:
                    current_idcg = (doc / math.log2(rank) + prev_idcg) if doc > 0 else prev_idcg
                prev_idcg = current_idcg
                idcg_values[rank] = current_idcg

            for rank in rel_rets:
                # Add the idcg key and its corresponding value directly to rel_rets here
                if rank <= len(idcg_values):
                    rel_rets[rank]['idcg'] = idcg_values.get(rank, 0.0)
                else:
                    rel_rets[rank]['idcg'] = idcg_values.get(len(idcg_values))

            n = len(rel_rets) if len(rel_rets) < n else n

            # Directly obtain the corresponding DCG and IDCG values and calculate NDCG
            dcg_n = rel_rets[n]['dcg']
            idcg_n = rel_rets[n]['idcg']

            ndcgs += (dcg_n / idcg_n) if idcg_n > 0 else 0
        return ndcgs / len(ret) if len(ret) > 0 else 0

    def map(self, rel, ret):
        """Calculate the MAP"""
        maps = 0
        for query_id in ret:
            rel_rets = self.get_rel_ret(rel, ret, query_id)
            rel_doc_nums = sum(1 for rel_doc in rel[query_id].values() if rel_doc > 0)  # Obtain the quantity of all relevant documents in rel
            query_id_map = 0
            inner_rank = 1  # Calculate the ranking of each related document in rel_ret
            for rank, doc in rel_rets.items():
                if doc['relevance'] > 0:
                    query_id_map += float(inner_rank / rank)
                    inner_rank += 1
            maps += query_id_map / rel_doc_nums
        return maps / len(ret) if len(ret) > 0 else 0

    """
    In small corpus, do not implement Bpref part.
    Because in the small corpus of qrels.txt, all documents are related, which means that the calculation results of Bpref and Recall are the same.
    Ignore Bpref here, but implement Bpref in a large corpus
    """

    def evaluate(self):
        start_time = time.time()
        print("Evaluation results:")

        precision = self.precision(self.rel, self.ret)
        recall = self.recall(self.rel, self.ret)
        r_precision = self.r_precision(self.rel, self.ret)
        p_15 = self.precision_15(self.rel, self.ret)
        ndcg_15 = self.ndcg_15(self.rel, self.ret)
        map = self.map(self.rel, self.ret)

        end_time = time.time()
        duration = end_time - start_time

        print(f"{'Precision:':<12} {precision:.4f}")
        print(f"{'Recall:':<12} {recall:.4f}")
        print(f"{'R-Precision:':<12} {r_precision:.4f}")
        print(f"{'P@15:':<12} {p_15:.4f}")
        print(f"{'NDCG@15:':<12} {ndcg_15:.4f}")
        print(f"{'Map:':<12} {map:.4f}")
        print(f"Evaluation cost {duration} s...")
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate_Small_corpus")
    parser.add_argument("-p", "--path", type=str, help="Path to the small results file...")

    args = parser.parse_args()

    qrels_path = os.path.join(args.path, 'files', 'qrels.txt')
    results_path = os.path.join(os.getcwd(), '22207296-small.results')

    if not os.path.exists(qrels_path):
        print("Small qrels file does not exist...")
    elif not os.path.exists(results_path):
        print("Small results file does not exist...")
    else:
        evaluator = Evaluator(qrels_path, results_path)
        evaluator.evaluate()
