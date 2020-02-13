
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
import librosa as li
import os
def vocoder_callback(*args,**kwargs):
    pass

input_audio_path = Path("genrated_audio")
dir_list = os.listdir(input_audio_path)
wav = li.load(str(input_audio_path/dir_list[0]),sr=44100,mono=True)
encoder_wav = encoder.preprocess_wav(wav)
embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)


model_dir = Path("synthesizer")/"saved_models"/"logs-pretrained"
checkpoints_dir = model_dir.joinpath("taco_pretrained")
synthesizer = Synthesizer(checkpoints_dir, low_mem=True)

texts = [""]

embeds = np.stack([embed] * len(texts))
specs = synthesizer.synthesize_spectrograms(texts, embeds)
breaks = [spec.shape[1] for spec in specs]
spec = np.concatenate(specs, axis=1)

out_wav = vocoder.infer_waveform(spec, progress_callback=vocoder_callback)
write("generated_audio/test.wav",44100,out_wav)