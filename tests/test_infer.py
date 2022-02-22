import pytest
import torch

from .test_model import MockModel
from .test_data import get_mock_dataset, get_mock_processed_dataset

from secora.infer import *
from secora.data import get_loader


def test_build_embedding():
    m = MockModel(128)
    dataset = get_mock_processed_dataset(mlen=100)
    dl = get_loader(dataset, batch_size=1)
    #split = preproces
    d_embs = build_embedding_space(m, dl, feature_prefix='func_')
    c_embs = build_embedding_space(m, dl, feature_prefix='code_')
    assert d_embs.shape == (100, 128)
    assert c_embs.shape == (100, 128)
    
