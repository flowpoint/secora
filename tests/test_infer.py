import pytest
import torch

from secora.infer import *
from secora.data import get_loader
from secora.display import Display


@pytest.mark.embedding_size(128)
@pytest.mark.token_per_sample(100)
def test_build_embedding(mock_model, mock_processed_dataset):
    m = mock_model #MockModel(128)
    dataset = mock_processed_dataset
    dl = get_loader(dataset, batch_size=1)
    #split = preproces
    embedding_size = 128
    d = Display(True)
    d_embs = build_embedding_space(m, dl, embedding_size, feature_prefix='func_', display=d)
    c_embs = build_embedding_space(m, dl, embedding_size, feature_prefix='code_', display=d)
    assert d_embs.shape == (100, 128)
    assert c_embs.shape == (100, 128)
    
