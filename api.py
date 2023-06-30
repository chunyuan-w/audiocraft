
import time
import os
os.environ['CURL_CA_BUNDLE'] = ''

import torch

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=30)  # generate 30 seconds.

model.lm.transformer = torch.quantization.quantize_dynamic(
    model.lm.transformer, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
)
model.lm.linears = torch.quantization.quantize_dynamic(
    model.lm.linears, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
)

#wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
#descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
descriptions = ['happy rock']
t0 = time.time()
wav = model.generate(descriptions)  # generates 3 samples. 1 text to music
print("generate time: ", time.time() - t0)
#melody, sr = torchaudio.load('./assets/bach.mp3') # text+melody to music
# generates using the melody from the given audio and the provided descriptions.
#wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr) # text+melody to music

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
