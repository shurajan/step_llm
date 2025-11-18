FROM python:3.6-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    openssh-server \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# SSH
RUN mkdir -p /run/sshd /root/.ssh && \
    echo "root:123456" | chpasswd && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

# Инициализация bash
RUN echo 'export PATH="/usr/local/bin:$PATH"' >> /root/.bashrc

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install numpy==1.16.6
RUN pip install torch==1.7.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install dill==0.3.4

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
