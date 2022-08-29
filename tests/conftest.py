# content of conftest.py

import pytest
import torch
from datasets import Dataset

'''
by default run all testcombinations, just to be sure
'''

def pytest_addoption(parser):
    parser.addoption(
        "--fastonly", action="store_true", default=False, help="don't run slow tests"
    )

    parser.addoption(
        "--nocuda", action="store_true", default=False, help="don't run cuda dependent tests"
    )

    # run only the heaviest tests (model+gpu test)
    # invoke sequentially
    parser.addoption(
        "--nosimple", action="store_true", default=False, help="don't run unmarked simple tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "cuda: marks tests to require cuda")


def pytest_collection_modifyitems(config, items):
    complex_markers = []

    if config.getoption("--fastonly"):
        skip_slow = pytest.mark.skip(reason="--fastonly option is true, skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    complex_markers.append('cuda')
    if config.getoption("--nocuda"):
        skip_cuda = pytest.mark.skip(reason="--nocuda is true, skipping cuda tests")

        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


    if config.getoption("--nosimple"):
        skip_simple = pytest.mark.skip(reason="--nosimple is true, skipping simple tests")

        for item in items:
            if not any([marker in item.keywords for marker in complex_markers]):
                item.add_marker(skip_simple)


class MockModel(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.embedding_size = emb
        self.state = {'key1':"value1"}

    def state_dict(self):
        return self.state

    def load_state_dict(self, d):
        self.state = d

    def forward(self, *args, **kwargs):
        return torch.ones([1,128], dtype=torch.float32)

@pytest.fixture
def mock_model(request):
    embsize = request.node.get_closest_marker("embedding_size").args[0]
    if embsize is None:
        raise RuntimeError('specify embedding_size as fixture marker')

    return MockModel(embsize)


@pytest.fixture
def mock_dataset(request):
    toks = request.node.get_closest_marker("token_per_sample").args[0]

    if toks is None:
        raise RuntimeError('specify token_per_sample as fixture marker')

    d = {"url": ['u']*toks, 
        'func_documentation_tokens':['a']*toks, 
        'func_code_tokens': ['b']*toks,
        "language": ['python']*toks}
    return Dataset.from_dict(d)

@pytest.fixture
def mock_processed_dataset(request):
    toks = request.node.get_closest_marker("token_per_sample").args[0]

    if toks is None:
        raise RuntimeError('specify token_per_sample as fixture marker')

    d = {"url": ['u']*toks, 
        'func_input_ids': [1]*toks, 
        'func_attention_mask': [1]*toks,
        'code_input_ids': [1]*toks, 
        'code_attention_mask': [1]*toks,
        "language": ['python']*toks}
    return Dataset.from_dict(d)


@pytest.fixture
def mock_train_split():
    ds = get_mock_dataset()
    conf = {'preprocess_cores': 1, 'languages': ['python'], 'max_input_tokens': 256, 'model_name': 'microsoft/codebert-base', 'preprocess_mode': 'concat'}
    return preprocess_split('train', conf)

