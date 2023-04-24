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
&& pip install ipython \
&& pip install ipykernel \
&& pip install scipy \
&& pip install pandas \
&& pip install matplotlib \
&& pip install -U scikit-learn \
&& pip install tensorboard \
&& pip install PyYAML

RUN pip install torcheval \
&& pip install torchsummary \
&& pip install progressbar \
&& pip install torchmetrics \
&& pip install pytorch-lightning
  

RUN git config --global user.name ${USERNAME} \
&& git config --global user.email kstefanidis48@gmail.com