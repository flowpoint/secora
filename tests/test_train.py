import pytest
import torch

from secora.models import *

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


