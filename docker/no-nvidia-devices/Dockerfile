FROM nvidia/cuda:11.7.1-base-ubuntu22.04

RUN apt update && apt install -y \
    ca-certificates \
    curl \
    git \
    make \
    parallel \
    python3 \
    python3-pip \
    sudo \
    zsh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN adduser --disabled-password --gecos '' --shell /bin/zsh user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

ENV HOME=/home/user
ENV PYTHONPATH /work

WORKDIR /work

# dotfiles
RUN export GIT_SSL_NO_VERIFY=1 \
    && mkdir -p ${HOME}/dotfiles \
    && git clone https://github.com/atahatah/dotfiles.git ${HOME}/dotfiles \
    && cd ${HOME}/dotfiles \
    && make install

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD [ "/bin/zsh" ]