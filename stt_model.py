import base64
import io
import json
import os
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Text

import asyncio
from aiohttp import ClientSession
from pydub import AudioSegment, silence

from utils import process_audio

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    pass


WIT_KEY_ACCESS_TOKENS = json.loads(os.getenv('WIT_KEY'))
WIT_KEY_ACCESS_TOKENS = cycle(WIT_KEY_ACCESS_TOKENS)


def split_into_chunks(
    segment: List,
    length: Optional[float] = 20000 / 1001,
    split_on_silence: Optional[bool] = False,
    noise_threshold: Optional[float] = -36
) -> List[AudioSegment]:
    """
    Splits an array wih audio signal into chunks

    Parameters
    ----------
    segment : list
        array with audio signal
    length : float, optional
        length of each chunk, by default 20000/1001
    split_on_silence : bool, optional
        wether to split audio when silence is identified, by default False
    noise_threshold : float, optional
        noise threshold denoting silence, by default -36

    Returns
    -------
    list
        chunks of audio
    """
    chunks = list()

    if split_on_silence is False:
        for i in range(0, len(segment), int(length * 1000)):
            chunks.append(segment[i:i + int(length * 1000)])
    else:
        while len(chunks) < 1:
            chunks = silence.split_on_silence(segment, noise_threshold)
            noise_threshold += 4

    for i, chunk in enumerate(chunks):
        if len(chunk) > int(length * 1000):
            subchunks = split_into_chunks(
                chunk, length, split_on_silence, noise_threshold + 4
            )
            chunks = chunks[:i - 1] + subchunks + chunks[i + 1:]

    return chunks


def read_audio_into_chunks(
    file_path: Text,
    length: Optional[float] = 20000 / 1001,
    split_on_silence: Optional[bool] = False,
    noise_threshold: Optional[float] = -36,
    sample_width: Optional[int] = 2,
    channels: Optional[int] = 1,
    frame_rate: Optional[int] = 48000
) -> List[AudioSegment]:
    """
    Loads audio from file, preprocessess and splits into chunks

    Parameters
    ----------
    file_path : str
        path to audio file
    length : float, optional
        length of each chunk, by default 20000/1001
    split_on_silence : bool, optional
        wether to split audio when silence is identified, by default False
    noise_threshold : float, optional
        noise threshold denoting silence, by default -36
    sample_width : int, optional
        new sample width of audio, by default 2
    channels : int, optional
        new amount of channels of audio, by default 1
    frame_rate : int, optional
        new frame rate of audio, by default 48000

    Returns
    -------
    list
        chunks of processed audio
    """
    audio = AudioSegment.from_file(file_path, format='ogg')
    audio = process_audio(
        audio,
        remove_silence=False,
        sample_width=sample_width,
        channels=channels,
        frame_rate=frame_rate
    )
    return split_into_chunks(
        audio,
        length=length,
        split_on_silence=split_on_silence,
        noise_threshold=noise_threshold
    )


async def transcribe_chunk(
    session: ClientSession, url: Text, headers: Dict, data: bytes
) -> Text:
    async with session.post(
        url,
        headers=headers,
        data=io.BufferedReader(io.BytesIO(data)),
        raise_for_status=True
    ) as resp:
        resp_json = await resp.json()
        return resp_json.get('text') or ''


async def transcribe_audio_wit(file_path: Text) -> Text:
    """
    Transcribes audio from file into text

    Parameters
    ----------
    file_path : str
        path to audio file

    Returns
    -------
    str
        audio transcription
    """
    url = 'https://api.wit.ai/speech?v=20200513'

    authorization = 'Bearer ' + next(WIT_KEY_ACCESS_TOKENS)

    content_type = 'audio/raw;' \
        'encoding=signed-integer;' \
        'bits=16;' \
        'rate=48000;' \
        'endian=little'

    headers = {
        'authorization': authorization,
        'content-type': content_type
    }

    chunks = read_audio_into_chunks(file_path)

    async with ClientSession() as session:
        tasks = []
        for audio in chunks:
            tasks.append(
                asyncio.ensure_future(
                    transcribe_chunk(session, url, headers, audio.raw_data)
                )
            )
        texts = await asyncio.gather(*tasks)
        return ' '.join(texts)


async def transcribe_audio(encoded_audio: Text) -> Text:
    """
    Transcribes encoded audio

    Parameters
    ----------
    audio : str
        encoded audio

    Returns
    -------
    str
        audio transcription
    """
    if encoded_audio is None:
        return None
    decoded_string = base64.b64decode(encoded_audio)
    audio_file = io.BytesIO(decoded_string)
    audio_file.seek(0)

    return await transcribe_audio_wit(audio_file)
