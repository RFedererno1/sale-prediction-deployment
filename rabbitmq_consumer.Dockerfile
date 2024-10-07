FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r requirements.txt

COPY . .