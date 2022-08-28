# secora

## goal
Build a robust retrieval augmented code model setup and learn along the way.

### side goals
some scalability
hyper parameter search
track training metrics

### non goals
research and handcraft new model architecture components
extreme scalability
multimodal
sota scores

#### smart 
* specific: model and code that runs inference on one gpu that i consider using day to day
* measurable: do i still like use it after a week
* achievable: yes
* relevant: i think ppl. would like to use foss code search andor build ontop of such as a backbone
* time-bound: be done in half a year


## description

this experimental project investigates Bert trained on CodeSearchNet with the contrastive SimCSE setup.

the similar model Codebert was trained for 160 GPU hours.
Difficulty in implementing distributed computation, we could practically only train for a shorter period.

our latest results can be seen in `visualization/Screenshot from 2022-02-28 23-51-52.png`

our previous graph, trained on just the python subset of CodeSearchNet is `visualization/Screenshot from 2022-02-17 11-54-51.png`

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

## type check:
```
mypy secora/train.py
```

## run tests:
```
# run all test
pytest
# skip slow and cuda tests
pytest --fastonly --nocuda
```

## build with meson

if build_container is true, it saves the docker container to builddir
if you don't have enough ram/space, you need to change the builddir path

```
meson setup /tmp/builddir

# after some modifications you maybe also need
meson --reconfigure /tmp/builddir

# configure the build, see meson_options.txt
cd /tmp/builddir
meson configure -Dbuild_container=true
meson compile -v
```

## design choices
### why meson
i needed a general build tool
bash scripts are errorprone
setuptools don't seem to work well with multilanguage projects or build containers
meson has good syntax and features

### why flat project layout

simpler to use the repl for research, and flat hierarchies are generally better

### validation

ml training has high compute and time costs
validation is key to deliver progress
validation is required for reproducable and comparable research and ml models
i try to consider and explore each of four parts of a ml "deliverable"

* dataset and domain insights
* model training metrics and methodologies
* ml coding strategies and patterns (throught this repo)
* a runnable model for inference

validation happens through the ordered steps:

1. dependecy building/integration testing
2. mypy type checking
3. linting
4. config checking
5. running all tests/ including short training
6. packaging in docker
7. gradually scale the runs
8. monitor metrics/ hparam search and hparam search is robust against spurious failed runs
9. model evaluation/checklist


## background
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

the previous course submission can be found under the tag `submission_1`
