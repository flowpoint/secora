import pytest
import torch

from secora.losses import *


def test_contrastive_loss():
    emb1 = emb2 = torch.tensor([[1.,0.],[0, 1.]])
    loss = contrastive_loss(emb1, emb2)


def test_mrr():
    # example of wikipedia
    true_results = ['cats', 'tori', 'viruses']

    rankings = [
        ['catten', 'cati', 'cats'],
        ['torii', 'tori', 'toruses'],
        ['viruses', 'virii', 'viri'],
        ]

    r = mrr(true_results, rankings)
    assert r == pytest.approx(0.61, 0.2)
