FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt update && apt install -y \
    ca-certificates \
    curl \
    git \
    make \
    parallel \
    sudo \
    zsh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# dotfiles
RUN export GIT_SSL_NO_VERIFY=1 \
    && mkdir -p ${HOME}/dotfiles \
    && git clone https://github.com/atahatah/dotfiles.git ${HOME}/dotfiles \
    && cd ${HOME}/dotfiles \
    && make install

ENV PYTHONPATH /work
ENV PATH $PATH:/home/user/.local/bin

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD [ "/bin/zsh" ]