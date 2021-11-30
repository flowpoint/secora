from itertools import starmap
import numpy as np

# my testimplementation of mrr
def rr(relevant_id: str, ranking: [str]):
    if relevant_id not in ranking:
        return 0
    rrank = 1 / (ranking.index(relevant_id)+1)
    return rrank

rr(1, [1,2,3])

def mrr(true_ids: [str], rankings: [[str]]):
    return np.mean(list(starmap(rr, zip(true_ids, rankings))))


# example of wikipedia
trues = ['cats', 'tori', 'viruses']

rankings = [
    ['catten', 'cati', 'cats'],
    ['torii', 'tori', 'toruses'],
    ['viruses', 'virii', 'viri'],
    ]

mrr(trues, rankings)

