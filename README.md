# Agendador Voice Server

This is the server responsible for STT and TTS

The endpoins are:

```txt
/: Health check.
    request:
        GET /

/stt: Processes and audio file and outputs its transcription
    request:
        POST /stt
        {
            'audio': '<wav audio file in base64 format>'
        }
    response:
        {
            'text': '<audio transcription>'
        }

/tts: Synthesizes a text sentence into an audio file
    request:
        POST /tts
        {
            'tts': '<text to be synthesized>'
        }
    response:
        {
            'audio': '<wav audio file in base64 format>'
        }
```

Environment variables

- A list of keys for the wit.ai api: `WIT_KEY=["<key 1>", "<key 2>"]
- A google drive link with the model: `MODEL_LINK=<google-drive-id>

You can run this as a docker container

- `docker build --tag <image-name> .`
- `docker run -p 5025:5025 -e WEB_CONCURRENCY=<number-of-workers> <image-name>`
- If you want to update the model, add the `UPDATE_MODEL=1` variable

To run locally:

- clone the TTS repository with `git clone https://github.com/flych3r/TTS -b pt-br`
- install the api requirements: `pip install -r requirements.txt`.
- run: `bash start.sh`

In the voice interface, add `VOICE_URL=http://localhost:5025/{}` to the `.env` file.
