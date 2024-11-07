FROM nvidia/cuda:11.7.1-base-ubuntu22.04

RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
RUN apt update && apt install -y \
    ca-certificates \
    curl \
    git \
    make \
    parallel \
    python3 \
    python3-pip \
    sudo \
    vim \
    zsh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ARG USERNAME=user
ARG GROUPNAME=user
ARG UID=1000
ARG GID=1000

RUN groupadd -r -g ${GID} ${GROUPNAME} && \
    useradd -r -m -g ${GROUPNAME} -p "" -u ${UID} --shell /bin/zsh ${USERNAME} && \
    mkdir -p /home/${USERNAME}
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER ${USERNAME}

ENV HOME=/home/${USERNAME}
ENV PATH $PATH:/home/${USERNAME}/.local/bin
ENV PYTHONPATH /work

WORKDIR /work

# dotfiles
RUN export GIT_SSL_NO_VERIFY=1 \
    && mkdir -p ${HOME}/dotfiles \
    && git clone https://github.com/atahatah/dotfiles.git ${HOME}/dotfiles \
    && cd ${HOME}/dotfiles \
    && make install_locally

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt 

CMD [ "/bin/zsh" ]