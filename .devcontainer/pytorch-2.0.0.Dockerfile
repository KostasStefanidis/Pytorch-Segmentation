FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG OS_USER_ID
ARG OS_GROUP_ID
ARG USERNAME
ARG GIT_USER_EMAIL

#RUN apt install sudo=1.8.31-1ubuntu1 -y
RUN groupadd --gid ${OS_GROUP_ID} ${USERNAME} \
&& useradd --uid ${OS_USER_ID} --gid ${OS_GROUP_ID} --shell /bin/bash --create-home --no-user-group ${USERNAME} \
&& chown ${OS_USER_ID}:${OS_GROUP_ID} /mnt/

# install usefull linux packages
RUN apt-get update \
&& apt-get install git -y \
&& apt-get install zip -y \
&& apt-get install unzip -y \
&& apt-get install wget -y \
&& apt-get install curl -y \
&& apt-get install screen -y

USER ${OS_USER_ID}

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

# configure git
RUN git config --global user.name ${USERNAME} \
&& git config --global user.email ${GIT_USER_EMAIL}