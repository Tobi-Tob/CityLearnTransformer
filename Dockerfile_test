FROM anibali/pytorch:latest
USER root

RUN apt-get update
RUN apt-get install gcc -y
RUN pip install tinyscaler
RUN pip install numpy
RUN pip install SuperSuit
RUN pip install gym
RUN pip install PettingZoo
RUN pip install stable-baselines3

RUN pip install git+https://github.com/intelligent-environments-lab/CityLearn.git@v1.3.3

ADD . /app/

RUN git config --global --add safe.directory /app
RUN git config --global --add safe.directory '*'

WORKDIR /app
RUN python /app/training.py