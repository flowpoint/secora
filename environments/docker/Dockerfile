FROM docker.io/rocm/pytorch:latest

# this was adapted from https://github.com/buildbot/buildbot/blob/master/worker/Dockerfile

LABEL org.opencontainers.image.authors="flowpoint@protonmail.com"

ENV HF_HOME=/root/.cache/huggingface

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
    # needed for ray[tune] log sync
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

# install faiss:
#RUN git clone https://github.com/facebookresearch/faiss && cd faiss
#RUN cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=OFF -B build .

#RUN conda install -c pytorch faiss-cpu
RUN pip install faiss-cpu

#RUN mkdir /buildbot &&\
#RUN useradd -ms /bin/bash buildbot

#COPY ./config /root/.ssh/config

RUN pip install tensorboard datasets ipython transformers ray[tune] docopt torch-SM3
RUN pip install pipenv

#COPY . /usr/src/buildbot-worker
#COPY ./buildbot_rocm.tac /root/buildbot.tac

#RUN pip3 install /usr/src/buildbot-worker && \
    #chown -R buildbot /buildbot

#COPY . /root/secora

WORKDIR /root/secora
#USER buildbot

CMD ["/usr/bin/dumb-init", "twistd", "--pidfile=", "-ny", "buildbot.tac"]
