import pytest
import torch

from secora.model import *


class MockModel(torch.nn.Module):
    def __init__(self, embsize):
        super().__init__()
        self.embedding_size = embsize

    def forward(self, *args, **kwargs):
        return torch.ones([1, 2, self.embedding_size])


@pytest.fixture
def get_model_inputs():
    input_ids = torch.ones([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)

    inputs  = input_ids, attention_mask
    return inputs


@pytest.mark.slow
def test_embeddingmodel():
    embsize = 128

    model = EmbeddingModel(BaseModel.CODEBERT, embsize)
    input_ids = torch.ones([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)
    outputs = model(input_ids, attention_mask)
    print('hello')
    assert outputs.shape == torch.Size([1, embsize])
    # normalization of output vectors
    print(torch.mean(outputs))
    assert all(torch.mean(outputs, dim=-1) < 1.)


@pytest.mark.slow
def test_bi_embeddingmodelcuda():
    embsize = 128

    model = EmbeddingModelCuda(BaseModel.CODEBERT, embsize, AMP.FP16 ).to('cuda')
    input_ids = torch.ones([1,512], dtype=torch.int64).to('cuda')
    attention_mask = torch.zeros([1,512], dtype=torch.int64).to('cuda')
    outputs = model(input_ids, attention_mask)
    assert outputs.shape == torch.Size([1, embsize])
    assert all(torch.mean(outputs, dim=-1) < 1.)




