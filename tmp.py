test_audio_filepath = "/Data/thorsten-de/wavs/5d000c81c8e7c4817cbfd7c4b8738feb.wav"
test_audio_text = "Dieser Geruch, wenn jemand eine Clementine \u00f6ffnet!"
fastpitch_model_path = "<path_to_fastpitch_nemo_or_ckpt>"
fastpitch_model_path = "/tmp/FastPitch--v_loss=0.7020-epoch=999-last.ckpt"

from matplotlib.pyplot import imshow
from nemo.collections.tts.models import FastPitchModel
from matplotlib import pyplot as plt
import librosa

import numpy as np

print("loading fastpitch melspecs via generate_spectrogram")
if ".nemo" in fastpitch_model_path:
    spec_model = FastPitchModel.restore_from(fastpitch_model_path).eval().cuda()
else:
    spec_model = FastPitchModel.load_from_checkpoint(checkpoint_path=fastpitch_model_path).eval().cuda()
text = spec_model.parse(test_audio_text, normalize=False)
spectrogram = spec_model.generate_spectrogram(
  tokens=test_audio_text, 
  speaker=None,
)
# imshow(spectrogram, origin="lower")
# plt.show()