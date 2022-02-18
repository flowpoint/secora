FROM docker.io/rocm/pytorch:latest

LABEL org.opencontainers.image.authors="flowpoint@protonmail.com"
ENV HF_HOME=/root/.cache/huggingface
ENV MASTER_ADDR=localhost
ENV MASTER_PORT=17909
WORKDIR /root

RUN apt update && apt-get install -y \
    python3-dev \
    tmux \
    vim \
    build-essential \
    git \
    subversion \
    python3-dev \
    libffi-dev \
    libssl-dev \
    rsync \
    python3-setuptools \
    python3-pip \
    # Test runs produce a great quantity of dead grandchild processes.  In a
    # non-docker environment, these are automatically reaped by init (process 1),
    # so we need to simulate that here.  See https://github.com/Yelp/dumb-init
    dumb-init \
    curl && \
    # Install required python packages, and twisted
    pip3 --no-cache-dir install 'twisted[tls]' && \
    pip3 install virtualenv && \
    rm -rf /var/lib/apt/lists/* 

# choose a faiss installation path:
#RUN git clone https://github.com/facebookresearch/faiss && cd faiss
#RUN cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=OFF -B build .

RUN pip install faiss-cpu
RUN pip install pipenv pytest tensorboard datasets ipython transformers docopt optuna torch-SM3

WORKDIR /root/secora
