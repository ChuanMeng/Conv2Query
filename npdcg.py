import numpy as np
from typing import List, Dict, Tuple


def calculate_npdcg(retrieved: List[Dict[str, List[Tuple[str, int]]]], cutoffs: List[int]) -> Dict[int, float]:
    npdcg_values = {}

    for cutoff in cutoffs:
        pdcg = calculate_pdcg(retrieved, cutoff)
        ipdcg = calculate_ipdcg(retrieved, cutoff)
        npdcg = pdcg / ipdcg if ipdcg != 0 else 0
        npdcg_values[cutoff] = npdcg
    return npdcg_values

def calculate_pdcg(retrieved: List[Dict[str, List[Tuple[str, int]]]], cutoff: int) -> float:
    pdcg = 0
    Z = 0
    ideal_position_l = {}
    labels = {}

    for i, utterance in enumerate(retrieved): # how many dicts in the list
        for doc, label in utterance["correct_docs"]:
            if doc not in ideal_position_l and label > 0:
                ideal_position_l[doc] = i # bug: a doc might be annotated multiple times in a conversation. only its first occurrence will be recorded;
                labels[doc] = label

    checked_docs = set()
    for i, utterance in enumerate(retrieved): # {}
        retrieved_docs = utterance["retrieved_docs"][:cutoff] # a list
        # retrieved_docs are "[]" if not perfrom retreival
        if len(retrieved_docs) > 0:
            Z += 1 # the number of times of retrieval
            dcg = 0
            for j, doc in enumerate(retrieved_docs): # [docid, ...]
                if doc in ideal_position_l and i >= ideal_position_l[doc]:

                    if doc not in checked_docs: # do not score repetitive items; bug: only remove relevant docs
                        checked_docs.add(doc)
                        dcg += labels[doc] / np.log2(2 + i - ideal_position_l[doc]) / np.log2(j + 2) # +2 bacause of using index here instead of rank
            pdcg += dcg
    return pdcg / Z if Z != 0 else 0


def calculate_ipdcg(retrieved: List[Dict[str, List[Tuple[str, int]]]], cutoff: int) -> float:
    ipdcg = 0
    Z = 0
    ideal_position_l = {}
    labels = {}
    for i, utterance in enumerate(retrieved):
        for doc, label in utterance["correct_docs"]:
            if doc not in ideal_position_l and label > 0:
                ideal_position_l[doc] = i
                labels[doc] = label

    checked_docs = set()
    for i, utterance in enumerate(retrieved):
        retrieved_docs = [doc for doc, _ in sorted(utterance["correct_docs"], key=lambda x: x[1], reverse=True)][:cutoff]

        if len(retrieved_docs) > 0:
            Z += 1
            dcg = 0
            for j, doc in enumerate(retrieved_docs):
                if doc not in checked_docs:
                    checked_docs.add(doc)
                    rel = labels[doc]
                    dcg += rel / np.log2(j + 2)
            ipdcg += dcg
    return ipdcg / Z if Z != 0 else 0



if __name__ == '__main__':
    """
    retrieved = [
        {
            "retrieved_docs": [1, 2, 30, 4],
            "correct_docs": [(1, 2), (2, 1), (14, 2)]
        },
        {
            "retrieved_docs": [],
            "correct_docs": [(9, 1)]
        },
        {
            "retrieved_docs": [3, 7, 6, 9, 10],
            "correct_docs": [(3, 2)]
        },
        {
            "retrieved_docs": [12, 13, 14, 30],
            "correct_docs": []
        }
    ]
    """

    retrieved = [
        {
            "retrieved_docs": [1,2],
            "correct_docs": [(2, 2),(1, 1)]
        },
        {
            "retrieved_docs": [5,6],
            "correct_docs": [(3, 2),(2, 1)]
        },
        {
            "retrieved_docs": [],
            "correct_docs": [(7,1),(8,1)]
        }
    ]

    cutoffs = [5, 10, 20, 100]
    npdcg_values = calculate_npdcg(retrieved, cutoffs)
    print(npdcg_values)
