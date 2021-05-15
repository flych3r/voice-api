FROM python:3.8

USER root

WORKDIR /app/

RUN apt-get update -qq && apt-get install -y gcc g++ ffmpeg espeak
RUN pip install --upgrade pip

RUN git clone https://github.com/flych3r/TTS -b pt-br

RUN pip install --no-cache-dir -e TTS/

ARG MODEL_LINK
RUN gdown --id $MODEL_LINK -O tts.zip
RUN unzip -o tts.zip
RUN mkdir /app/TTS/models
RUN mv tts/* /app/TTS/models
RUN rm -rf tts tts.zip

COPY *.py /app/
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5025

COPY start.sh /app
RUN chmod +x /app/start.sh
RUN chmod -R 777 /app

RUN mkdir /.cache
RUN chmod -R 777 /.cache

ENV NUMBA_CACHE_DIR=/tmp/
ENV MPLCONFIGDIR=/tmp/

USER 1001

ENTRYPOINT ["/app/start.sh"]
