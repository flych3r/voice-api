from typing import Optional

from pydub import AudioSegment
from pydub.silence import split_on_silence


def process_audio(
    audio: AudioSegment,
    remove_silence: Optional[bool] = False,
    silence_thresh: Optional[int] = -36,
    sample_width: Optional[int] = 2,
    channels: Optional[int] = 1,
    frame_rate: Optional[int] = 48000
) -> AudioSegment:
    """
    Changes audio sample width, channels, frame rate and removes silence

    Parameters
    ----------
    audio : AudioSegment
        audio
    remove_silence : bool, optional
        wether to remove silence, by default False
    silence_thresh: int, optional
        noise threshold to consider as silence, by default -36
    sample_width : int, optional
        new sample width of audio, by default 2
    channels : int, optional
        new amount of channels of audio, by default 1
    frame_rate : int, optional
        new frame rate of audio, by default 48000

    Returns
    -------
    AudioSegment
        processed audio
    """
    if remove_silence:
        audio = sum(split_on_silence(audio, silence_thresh=silence_thresh))
    return audio.set_sample_width(
        sample_width
    ).set_channels(
        channels
    ).set_frame_rate(
        frame_rate
    )
