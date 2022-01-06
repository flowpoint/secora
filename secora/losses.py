from itertools import starmap
import numpy as np

import torch
import torch.nn.functional as F


def contrastive_loss(model, batch, config):
    ''' the loss used by simcse
    inspired by:
    https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
    '''

    input_ids = batch['input_ids'].to(config['device'], non_blocking=True)
    token_type_ids = batch['token_type_ids'].to(config['device'], non_blocking=True)
    attention_mask = batch['attention_mask'].to(config['device'], non_blocking=True)

    emb1 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    emb2 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    # use the exact same loss from the simcse repository
    sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1) / config['temp']
    labels = torch.arange(sim.size(0), dtype=torch.int64, device=config['device'])
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


