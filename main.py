from base64 import b64decode
from io import BytesIO
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, validator
from pydub import AudioSegment

from stt_model import transcribe_audio
from tts_model import synthesize_text

app = FastAPI(title='Voice Server', version='1.1.0')


class Status(BaseModel):
    """
    Server status

    Used to check if server is running.
    """
    status = 'ok'


class AudioJSON(BaseModel):
    audio: Optional[str]

    @validator('audio')
    def audio_must_be_ogg_and_not_exceed_duration(cls, value):
        if value:
            decoded_value = b64decode(value)
            file_type = decoded_value[:4].decode()

            if file_type != 'OggS':
                raise ValueError('Audio must be ogg.')

            audio_file = BytesIO(decoded_value)
            audio_file = AudioSegment.from_file_using_temporary_files(
                audio_file, format='ogg'
            )

            if audio_file.duration_seconds > 60:
                raise ValueError('Audio must not exceed 60 seconds.')
        return value


class TextJSON(BaseModel):
    text: Optional[str]

    @validator('text')
    def text_must_not_exceed_length(cls, value):
        if value:
            text_length = len(value)

            if text_length > 1250:
                raise ValueError('Text must not be bigger than 1250 characters.')
        return value


@app.get('/', response_model=Status)
async def health():
    return {'status': 'ok!'}


@app.post('/stt', response_model=TextJSON)
async def stt(input: AudioJSON) -> TextJSON:
    """
    Speech to Text

    Receives a base64 encoded ogg audio and returns it's transcription.
    """
    transcription = await transcribe_audio(input.audio)
    return {'text': transcription}


@app.post('/tts', response_model=AudioJSON)
async def tts(input: TextJSON) -> AudioJSON:
    """
    Text to Speech

    Receives a text and returns it's base64 encoded ogg audio synthesis.
    """
    synthesis = await synthesize_text(input.text)
    return {'audio': synthesis}
