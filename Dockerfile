FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="pelenczei.balint@sztaki.hun-ren.hu"
LABEL docker_image_name="MCTS Batch Optimization in Computer Vision"
LABEL description="This container is created to train and evaluate MCTS batch sequence optimized Computer Vision models"
LABEL com.centurylinklabs.watchtower.enable="true"

WORKDIR /mcts_batch_optimization_cv

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qy \
    python3-pip \
    build-essential \
    autoconf \
    automake \
    sudo \
    vim \
    nano \
    git \
    curl \
    wget \
    tmux \
    openssh-server

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

RUN mkdir /root/.ssh

EXPOSE 22

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

COPY . .

RUN pip3 install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/"
ENV PYTHONPATH "${PYTHONPATH}:/mcts_batch_optimization_cv/"

RUN echo "cd /mcts_batch_optimization_cv" >> /root/.bashrc

CMD ["/usr/sbin/sshd", "-D"]