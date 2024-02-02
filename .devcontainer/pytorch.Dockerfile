ARG VARIANT="2.2.0-cuda12.1-cudnn8-runtime"
FROM pytorch/pytorch:${VARIANT}

ARG USERNAME
RUN useradd --shell /bin/bash --create-home --no-user-group ${USERNAME}

# install usefull linux packages
RUN apt-get update \
&& apt-get install git -y \
&& apt-get install zip -y \
&& apt-get install unzip -y \
&& apt-get install wget -y \
&& apt-get install curl -y \
&& apt-get install screen -y

USER ${USERNAME}

# install usefull python packages
RUN pip install --no-cache-dir --upgrade pip \
&& pip install --no-cache-dir ipython==8.10.0 \
&& pip install --no-cache-dir ipykernel==6.22.0 \
&& pip install --no-cache-dir matplotlib==3.7.1 \
&& pip install --no-cache-dir -U scikit-learn==1.2.2 \
&& pip install --no-cache-dir PyYAML==6.0

# install pytorch extra packages
RUN pip install --no-cache-dir torcheval==0.0.6 \
&& pip install --no-cache-dir torchsummary==1.5.1 \
&& pip install --no-cache-dir torchmetrics==0.11.4 \
&& pip install --no-cache-dir lightning==2.0.2 \
&& pip install --no-cache-dir -U tensorboard-plugin-profile==2.11.2

# USER root
# RUN apt-get update && apt-get install -y libc6=2.28
