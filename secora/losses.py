from itertools import starmap
import numpy as np

import torch
import torch.nn.functional as F

import math

def contrastive_loss(emb1, emb2, temp: float=0.5):
    ''' InfoNCE loss function
    use the implementation from simcse:
    https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
    '''

    sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1) / temp
    labels = torch.arange(sim.size(0), dtype=torch.int64, device=sim.device)
    loss = F.cross_entropy(sim, labels)

    return loss


def rr(relevant_id: str, ranking: [str]):
    ''' reciprocal rank '''
    if relevant_id not in ranking:
        return 0
    rrank = 1 / (ranking.index(relevant_id)+1)
    return rrank


def mrr(true_ids: [str], rankings: [[str]]):
    ''' mean reciprocal rank '''
    r = list(starmap(rr, zip(true_ids, rankings)))
    return sum(r)/len(r)


def ndcg(lst):
    ''' normalized discounted cumulative gain '''

    score = 0
    ideal_score = 0
    ideal_list = sorted(lst, reverse=True)
    for i in range(0, len(lst)):
        print(i)
    score += lst[i]/math.log2(i+2)
    ideal_score += ideal_list[i]/math.log2(i+2)
    score = score/ideal_score
    return score

    #list=[2,3,3,1,2]
    #print(ndcg(list))
