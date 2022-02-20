# Secora

choose your environment:
- pipenv
- docker
- modules

### pipenv
```
# install pipenv:
pip install pipenv
# open the virtual environment:
pipenv shell
# install the needed packages in the virtual environment:
pipenv install
```

torch and faiss should be manually installed

```
# train the model with ipython:
ipython secora/train.py
```

## run tests:
```
pytest tests
```

## run in container
```
sudo ./environments/container/run.sh
python secora/train.py configs/default.yml --name run_py768_0 --progress
```

# Goals
we want to apply and evaluate bert on code for information retrieval.

- basically entry on leaderboard, with an information retrieval based model
- presentation
- working demo code
- (optional) blog post publication

non-goals or out of scope:

- writing our own paper 
- integration to other frameworks

open design decisions:
- how to do logging/ experiment tracking


important papers:  

- [code search net](https://arxiv.org/pdf/1909.09436.pdf)
- [Bert](https://arxiv.org/abs/1810.04805)
- [simCSE](https://arxiv.org/abs/2104.08821)
- [sBert](https://arxiv.org/abs/1908.10084)
- [codeBert](https://github.com/microsoft/CodeBERT)


leaderboard:

- [microsoft code challenges](https://microsoft.github.io/CodeXGLUE/)


implementation:

- [dataset](https://huggingface.co/datasets/code_x_glue_tc_text_to_code)
- [retrieval sBert library](https://www.sbert.net/index.html)
- [pytorch](https://pytorch.org/docs/stable/index.html)


other background information and literature:

- [Retrieval Augmented Coding](https://arxiv.org/pdf/2108.11601.pdf)
- [benchmark](https://github.com/openai/human-eval)
- [robustness analysis](https://arxiv.org/pdf/2002.03043.pdf)
- [collection of ai4code papers](https://github.com/bdqnghi/awesome-ai4code-papers)
- [evaluation of large models for code](https://arxiv.org/abs/2107.03374)
- [unsupervised code retrieval](https://arxiv.org/abs/2009.02731)
- [comprehensive literature review of the field](https://arxiv.org/abs/2009.06520)
- [codedotAI, open organization](https://github.com/CodedotAl)


existing software:

- [sourcegraph](https://sourcegraph.com/search)
- [openai codex](https://openai.com/blog/openai-codex/)


notes:
```
# setup
pip3 install pipenv
pipenv --python 3.9.2 shell
pipenv install

pip3 install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
