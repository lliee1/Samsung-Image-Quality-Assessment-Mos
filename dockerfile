FROM ufoym/deepo:all-py38-cu113

USER root

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get -y install libgl1-mesa-glx
RUN pip install --upgrade pip


WORKDIR /root
COPY . /root/dacon
WORKDIR /root/dacon/lightning-hydra-template

# requirements
RUN pip install -r requirements.txt
# warmup
RUN pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
