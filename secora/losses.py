from itertools import starmap
import numpy as np

import torch
import torch.nn.functional as F

import math

def contrastive_loss(emb1, emb2, temp: float=0.5):
    ''' the loss used by simcse
    inspired by:
    https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
    '''

    # use the exact same loss from the simcse repository
    sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1) / temp
    labels = torch.arange(sim.size(0), dtype=torch.int64, device=sim.device)
    loss = F.cross_entropy(sim, labels)

    return loss


def rr(relevant_id: str, ranking: [str]):
    ''' usage:
        rr(1, [1,2,3])
    '''

    if relevant_id not in ranking:
        return 0
    rrank = 1 / (ranking.index(relevant_id)+1)
    return rrank


def mrr(true_ids: [str], rankings: [[str]]):
    '''
    usage like:
    # example of wikipedia
    trues = ['cats', 'tori', 'viruses']

    rankings = [
        ['catten', 'cati', 'cats'],
        ['torii', 'tori', 'toruses'],
        ['viruses', 'virii', 'viri'],
        ]

    mrr(trues, rankings)
    '''

    return np.mean(list(starmap(rr, zip(true_ids, rankings))))




def ndcg(lst):
    ''' author kai '''

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
