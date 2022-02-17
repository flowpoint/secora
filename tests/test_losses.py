import pytest
import torch

from secora.losses import *

from .test_model import get_model_inputs

class MockModel:
    def __init__(self):
        self.state = {'key1':"value1"}
    def state_dict(self):
        return self.state
    def load_state_dict(self, d):
        self.state = d


class BiMockModel(torch.nn.Module):
    def __init__(self):
        super(BiMockModel, self).__init__()

    def forward(self, *args, **kwargs):
        return torch.ones([1,128], dtype=torch.float32)
        #return self.p*args[0][:128]


def test_contrastive_loss(get_model_inputs):
    m = BiMockModel()

    inputs = get_model_inputs
    biemb = m(*inputs)

    emb1 = biemb[:,0]
    emb2 = biemb[:,1]

    # use the exact same loss from the simcse repository
    sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1) /0.05

    labels = torch.arange(sim.size(0), dtype=torch.int64, device=sim.device)
    #print(emb1.dtype)
    #contrastive_loss(emb1, emb2, 0.05)
