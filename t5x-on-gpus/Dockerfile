ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.09-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN  apt-get update -y && apt-get install dnsutils -y

# Install the latest jax
RUN pip install --upgrade pip
RUN pip install -U  sentencepiece rouge_score t5

RUN pip install jax[cuda]==0.4.4 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /
RUN git clone --branch=main https://github.com/google-research/t5x
WORKDIR t5x
RUN pip install -e '.[gpu]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /scripts
ADD entrypoint.sh /scripts
ADD configs /configs/

ENV PYTHONPATH=/t5x

ENTRYPOINT ["/scripts/entrypoint.sh"]