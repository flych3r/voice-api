import base64
import hashlib
import re
import textwrap
import time
from pathlib import Path
from typing import Text

try:
    from pydub import AudioSegment
    import torch
    import numpy as np

    from utils import process_audio

    from TTS.tts.utils.generic_utils import setup_model
    from TTS.vocoder.utils.generic_utils import setup_generator
    from TTS.utils.io import load_config
    from TTS.tts.utils.text.symbols import symbols, phonemes
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.utils.synthesis import text_to_seqvec, numpy_to_torch
    from TTS.tts.utils.text.cleaners import phoneme_cleaners
    LIVE_TTS = True
    model_path = Path('/app/TTS/models/')
except ModuleNotFoundError:
    LIVE_TTS = False

output_dir = Path('/app') / 'voice_response'
output_dir.mkdir(parents=True, exist_ok=True)

tts_cache = Path('/.cache/tts')
tts_cache.mkdir(parents=True, exist_ok=True)


def model_inference(model, vocoder_model, text, CONFIG, use_cuda):
    inputs = text_to_seqvec(text, CONFIG)
    inputs = numpy_to_torch(inputs, torch.long, cuda=use_cuda)
    inputs = inputs.unsqueeze(0)

    if inputs.size()[1] == 0:
        return None

    _ , postnet_output, _, _ = model.inference(
        inputs, speaker_ids=None
    )
    postnet_output = postnet_output[0].data.cpu().numpy()
    waveform = vocoder_model.inference(
        torch.FloatTensor(postnet_output.T).unsqueeze(0)
    )
    waveform = waveform.flatten()
    if use_cuda or not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()
    return waveform


def load_tacotron2(use_cuda):
    """
    Loads the Tacotron2 model

    Parameters
    ----------
    use_cuda : bool
        whether to use the gpu

    Returns
    -------
    model, audio processor, model config
    """
    TTS_MODEL = model_path / 'model.pth.tar'
    TTS_CONFIG = model_path / 'config.json'

    TTS_CONFIG = load_config(TTS_CONFIG)
    TTS_CONFIG.audio['stats_path'] = str(model_path / 'scale_stats.npy')

    ap = AudioProcessor(**TTS_CONFIG.audio)

    num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
    model = setup_model(num_chars, 0, TTS_CONFIG)
    cp = torch.load(TTS_MODEL, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model'])
    if use_cuda:
        model.cuda()

    if 'r' in cp:
        model.decoder.set_r(cp['r'])

    model.eval()

    return model, ap, TTS_CONFIG


def load_vocoder(use_cuda):
    """
    Loads the Vocoder model

    Parameters
    ----------
    use_cuda : bool
        whether to use the gpu

    Returns
    -------
    model, audio processor, model config
    """
    VOCODER_MODEL = model_path / 'vocoder_model.pth.tar'
    VOCODER_CONFIG = model_path / 'vocoder_config.json'

    VOCODER_CONFIG = load_config(VOCODER_CONFIG)
    VOCODER_CONFIG.audio['stats_path'] = str(model_path / 'vocoder_scale_stats.npy')

    ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])

    vocoder_model = setup_generator(VOCODER_CONFIG)
    cp = torch.load(VOCODER_MODEL, map_location=torch.device('cpu'))
    vocoder_model.load_state_dict(cp['model'])
    vocoder_model.remove_weight_norm()
    vocoder_model.inference_padding = 0

    if use_cuda:
        vocoder_model.cuda()
    vocoder_model.eval()

    return vocoder_model, ap_vocoder, VOCODER_CONFIG


def load_tts_model():
    """
    Loads the Tacotron2 e Vocoder models

    Returns
    -------
    tacotron2 model, audio processor and configuration
    vocoder model, audio processor and configuration
    wether the gpu is being used
    """
    use_cuda = torch.cuda.is_available()
    model, ap, config = load_tacotron2(use_cuda)
    vocoder_model, ap_vocoder, vocoder_config = load_vocoder(use_cuda)
    return model, ap, config, vocoder_model, ap_vocoder, vocoder_config, use_cuda


def tts(fp, sentence, silence_duration=300):
    """
    Uses the models to create the audio file

    Parameters
    ----------
    fp : str
        path to save the audio file
    sentence : str
        sentence to synthesize
    """
    start = time.time()
    br = 240
    wavs = []
    length = 0
    for sent in filter(lambda x: x, re.split(r'[.|;|!|?|\n]', sentence)):
        for frase in textwrap.wrap(sent, width=br, break_long_words=False):
            cached = hashlib.sha1(str.encode(frase)).hexdigest()
            cached = tts_cache / '{}.wav'.format(cached)
            if cached not in tts_cache.glob('*.wav'):
                waveform = model_inference(
                    model, vocoder_model, ' ' + frase + ' ', config, use_cuda
                )
                if waveform is None:
                    continue
                length += len(waveform)
                ap.save_wav(waveform, cached)
            wavs.append(cached)
    wavs = [
        AudioSegment.from_wav(w)
        + AudioSegment.silent(silence_duration)
        for w in wavs
    ]
    waveform = sum(wavs)
    if len(waveform):
        waveform = process_audio(waveform, remove_silence=False)
        waveform.export(fp, format='ogg')
        end = time.time()
        total = end - start
        rtf = total / (length / ap.sample_rate)
        print(' > Run-time: {:.03f} seconds'.format(total))
        print(' > Real-time factor: {:.03f} x'.format(rtf))


if LIVE_TTS:
    (
        model, ap, config,
        vocoder_model, ap_vocoder, vocoder_config,
        use_cuda
    ) = load_tts_model()
    use_gl = False


async def synthesize_text(sentence: Text) -> Text:
    """
    Synthesizes a text sentence into audio

    Parameters
    ----------
    sentence : str
        sentence to be synthesized

    Returns
    -------
    str
        base64 encoded string with audio
    """
    if sentence is None:
        return None
    sentence = re.sub(r'Protocolo: \w{8}-\w{4}-\w{4}-\w{4}-\w{12}', ' ', sentence)
    fp = re.sub(r'\W', '', sentence.lower())
    if not fp:
        return None
    fp = hashlib.sha1(str.encode(fp)).hexdigest()
    fp = output_dir / '{}.ogg'.format(fp)
    if fp not in output_dir.glob('*.ogg'):
        if not LIVE_TTS:
            return None
        sentence = phoneme_cleaners(sentence)
        tts(fp, sentence)
    with open(fp, 'rb') as audio_file:
        encoded_bytes = base64.b64encode(audio_file.read())
        encoded_string = encoded_bytes.decode()
    return encoded_string
