FROM docker.io/rocm/pytorch:latest
#FROM docker.io/rocm/pytorch-nightly:latest

USER root
ENV HF_HOME=/root/.cache/huggingface
WORKDIR /root

RUN apt-get update && apt-get install -y \
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
    #ninja-build \
    # Test runs produce a great quantity of dead grandchild processes.  In a
    # non-docker environment, these are automatically reaped by init (process 1),
    # so we need to simulate that here.  See https://github.com/Yelp/dumb-init
    dumb-init \
    curl && \
    rm -rf /var/lib/apt/lists/* 
    # Install required python packages, and twisted
    #pip3 --no-cache-dir install 'twisted[tls]' && \
    #pip3 install virtualenv && \

# choose a faiss installation path:
#RUN git clone https://github.com/facebookresearch/faiss && cd faiss
#RUN cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=OFF -B build .

RUN pip install pipenv faiss-cpu pytest pytest-xdist pytest-cov tensorboard datasets ipython transformers docopt optuna torch-SM3 more_itertools onnx onnxruntime meson==0.62.2 ninja #jschon

# temporarily needed bc of missing jschon py3.7 support
RUN git clone --recursive https://github.com/flowpoint/jschon-backport.git /root/jschon-backport && cd /root/jschon-backport && python setup.py install && cd

RUN git clone https://github.com/luyug/GradCache && cd GradCache && pip install --editable .

ENV CONTAINER_RUNTIME="podman"
ENV MASTER_ADDR=127.0.0.1
ENV MASTER_PORT=9000

WORKDIR /root/secora
