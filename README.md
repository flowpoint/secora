# Secora

This experimental project investigates Bert trained on CodeSearchNet with the contrastive SimCSE setup.

The similar model Codebert was trained for 160 GPU hours.
Difficulty in implementing distributed computation, we could practically train for a shorter period.

our latest results can be seen in [here](visualization/Screenshot from 2022-02-28 23-51-52.png):
our previous graph, trained on just the python subset of CodeSearchNet is [here]('visualization/Screenshot from 2022-02-17 11-54-51.png)


## environments

- pipenv 
- docker (amd gpu)

## pipenv

```
# install pipenv:
pip install pipenv
# open the virtual environment:
pipenv shell
# install the needed packages in the virtual environment:
pipenv install

python -m secora.train configs/default.yml --progress --name distilroberta
```

## run in container
```
sudo ./environments/container/run.sh

python -m secora.train configs/default.yml --progress --name distilroberta
```

## show training curves
```
tensorboard --logdir ouput
```

## evaluate 
```

```

## run tests:
```
pytest
pytest --runslow --runcuda
```

important papers:  

- [code search net](https://arxiv.org/pdf/1909.09436.pdf)
- [simCSE](https://arxiv.org/abs/2104.08821)
- [codeBert](https://github.com/microsoft/CodeBERT)

challenge:

- [codesearchnet challenge](https://github.com/github/codesearchnet)


implementation:

- [dataset](https://huggingface.co/datasets/code_x_glue_tc_text_to_code)
- [pytorch](https://pytorch.org/docs/stable/index.html)
- [faiss](https://faiss.ai/)

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
