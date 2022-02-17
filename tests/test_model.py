import pytest
import torch

from secora.model import *

@pytest.fixture
def get_model_inputs():
    input_ids = torch.ones([1,512], dtype=torch.int64)
    token_type_ids = torch.zeros([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)

    inputs  = input_ids, token_type_ids, attention_mask
    return inputs


@pytest.mark.slow
def test_embeddingmodel():
    embsize = 128

    model = EmbeddingModel(BaseModel.CODEBERT, embsize)
    input_ids = torch.ones([1,512], dtype=torch.int64)
    token_type_ids = torch.zeros([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)
    outputs = model(input_ids, token_type_ids, attention_mask)
    print('hello')
    assert outputs.shape == torch.Size([1, embsize])
    # normalization of output vectors
    print(torch.mean(outputs))
    assert all(torch.mean(outputs, dim=-1) < 1.)

@pytest.mark.slow
def test_bi_embeddingmodel():
    embsize = 128

    model = BiEmbeddingModel(BaseModel.CODEBERT, embsize)
    input_ids = torch.ones([1,512], dtype=torch.int64)
    token_type_ids = torch.zeros([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)
    outputs = model(input_ids, token_type_ids, attention_mask)
    assert outputs.shape == torch.Size([1, 2, embsize])
    assert all(torch.mean(outputs[:, 0], dim=-1) < 1.)
    assert all(torch.mean(outputs[:, 1], dim=-1) < 1.)


@pytest.mark.slow
def test_bi_embeddingmodelcuda():
    embsize = 128

    model = BiEmbeddingModelCuda(BaseModel.CODEBERT, embsize, Precision.FP16 ).to('cuda')
    input_ids = torch.ones([1,512], dtype=torch.int64).to('cuda')
    token_type_ids = torch.zeros([1,512], dtype=torch.int64).to('cuda')
    attention_mask = torch.zeros([1,512], dtype=torch.int64).to('cuda')
    outputs = model(input_ids, token_type_ids, attention_mask)
    assert outputs.shape == torch.Size([1, 2, embsize])
    assert all(torch.mean(outputs[:,0], dim=-1) < 1.)
    assert all(torch.mean(outputs[:,1], dim=-1) < 1.)



