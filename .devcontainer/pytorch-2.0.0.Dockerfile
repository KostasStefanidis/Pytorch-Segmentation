FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG USERNAME
ARG USER_ID
ARG GROUP_ID

#RUN apt install sudo=1.8.31-1ubuntu1 -y
RUN groupadd --gid ${GROUP_ID} $USERNAME
RUN useradd --uid ${USER_ID} --gid ${GROUP_ID} --shell /bin/bash --create-home --no-user-group ${USERNAME}
RUN chown ${USER_ID}:${GROUP_ID} /mnt/

# install usefull linux packages
RUN apt update \
&& apt install git -y \
&& apt install zip -y \
&& apt install unzip -y \
&& apt install wget -y \
&& apt install curl -y \
&& apt install screen -y

USER ${USER_ID}

# install usefull python packages
RUN pip install --upgrade pip \
&& pip install ipython==8.10.0 \
&& pip install ipykernel==6.22.0 \
&& pip install pandas==2.0.0 \
&& pip install matplotlib==3.7.1 \
&& pip install -U scikit-learn==1.2.2 \
#&& pip install tensorboard \
&& pip install PyYAML==6.0

RUN pip install torcheval==0.0.6 \
&& pip install torchsummary==1.5.1 \
&& pip install torchmetrics==0.11.4 \
&& pip install lightning==2.0.2 \
&& pip install -U tensorboard-plugin-profile==2.11.2
  

RUN git config --global user.name ${USERNAME} \
&& git config --global user.email kstefanidis48@gmail.com