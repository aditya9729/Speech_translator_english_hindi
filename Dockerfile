FROM python:3.7.6

WORKDIR /app

# install requirements first for caching
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN apt-get update \
        && apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile1-dev -y \
        && pip3 install pyaudio
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY . /app

RUN mkdir ./data

RUN python ./src/download_models.py

CMD python app.py