# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import math
import os
import random
from typing import Callable, Dict, Iterable, List, Optional, Union

import braceexpand
import numpy as np
import torch
import webdataset as wd
from torch.utils.data import ChainDataset

from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging

__all__ = [
    'AudioToAudioDataset',
    'DynamicTargetAudioToAudioDataset',
    'AudioToSourceDataset',
    'StaticTargetAudioToAudioDataset',
]


class _AudioDataset(Dataset):
    """
    """

    def __init__(
        self,
        manifest_filepath: str,
        num_sources: int,
        featurizer,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(',')

        self.collection = collections.SpeechSeparationAudio(
            manifest_files=manifest_filepath,
            num_sources=num_sources,
            min_duration=min_duration,
            max_duration=max_duration,
            max_utts=max_utts,
        )

        self.num_sources = num_sources
        self.featurizer = featurizer
        self.orig_sr = kwargs.get('orig_sr', None)

    def get_manifest_sample(self, sample_id):
        return self.collection[sample_id]

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        features_list = [
            self.featurizer.process(
                sample.audio_file[i], offset=sample.offset[i], duration=sample.duration[i], orig_sr=self.orig_sr,
            )
            for i in range(self.num_sources)
        ]

        scale_factors = [sample.scale_factor[i] for i in range(self.num_sources)]
        return {
            'features_list': features_list,
            'scale_factors':  scale_factors
        }


class AudioToSourceDataset(_AudioDataset):
    """
    AudioToAudioDataset is intended for Audio to Audio tasks such as speech separation,
    speech enchancement, music source separation etc.
    """

    def __init__(
        self,
        manifest_filepath: str,
        num_sources: int,
        featurizer,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            num_sources=num_sources,
            featurizer=featurizer,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            **kwargs,
        )

        self.num_sources = num_sources

    def __getitem__(self, index):
        data_pt = super().__getitem__(index)
        features_list = data_pt['features_list']
        features_lengths = [torch.tensor(x.shape[0]).long() for x in features_list]

        min_l = torch.min(torch.stack(features_lengths)).item()

        if self.num_sources == 2:
            t1, t2 = [x[:min_l] for x in features_list]
            mix = t1 + t2
            output = [mix, torch.tensor(min_l).long(), t1, t2]
        elif self.num_sources == 3:
            t1, t2, t3 = [x[:min_l] for x in features_list]
            mix = t1 + t2 + t3
            output = [mix, torch.tensor(min_l).long(), t1, t2, t3]

        # add index
        output.append(index)

        return output

    def _collate_fn(self, batch):
        return _audio_to_audio_collate_fn(batch)


def _audio_to_audio_collate_fn(batch):
    """collate batch 
    Args:
        batch 
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, audio_lengths, _, _, sample_ids = packed_batch
    elif len(packed_batch) == 6:
        _, audio_lengths, _, _, _, sample_ids = packed_batch
    else:
        raise ValueError("Expects 5 or 6 tensors in the batch!")

    # convert sample_ids to torch
    sample_ids = torch.tensor(sample_ids, dtype=torch.int32)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signal, target1, target2, target3 = [], [], [], []
    for b in batch:
        if len(b) == 5:
            sig, sig_len, t1, t2, _ = b
        else:
            sig, sig_len, t1, t2, t3, _ = b
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
                t1 = torch.nn.functional.pad(t1, pad)
                t2 = torch.nn.functional.pad(t2, pad)
                if len(b) == 6:
                    t3 = torch.nn.functional.pad(t3, pad)

            audio_signal.append(sig)
            target1.append(t1)
            target2.append(t2)
            if len(b) == 6:
                target3.append(t3)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        target1 = torch.stack(target1)
        target2 = torch.stack(target2)
        if len(b) == 6:
            target3 = torch.stack(target3)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, target1, target2, target3, audio_lengths, sample_ids = None, None, None, None, None, None

    if len(packed_batch) == 5:
        return audio_signal, audio_lengths, target1, target2, sample_ids
    else:
        return audio_signal, audio_lengths, target1, target2, target3, sample_ids


class DynamicTargetAudioToAudioDataset(_AudioDataset):
    """
    AudioToAudioDataset is intended for Audio to Audio tasks such as speech separation,
    speech enchancement, music source separation etc.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports"""
        output = {}
        output['mixed_audio'] = NeuralType(('B', 'T'), AudioSignal())
        output['mixed_audio_len'] = NeuralType(tuple('B'), LengthsType())
        output['target1'] = NeuralType(('B', 'T'), AudioSignal())
        output['target2'] = NeuralType(('B', 'T'), AudioSignal())
        output['enrollment'] = NeuralType(('B', 'T'), AudioSignal())
        output['enrollment_len'] = NeuralType(tuple('B'), LengthsType())
        return output

    def __init__(
        self,
        manifest_filepath: str,
        featurizer,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            num_sources=1,
            featurizer=featurizer,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            **kwargs,
        )

    def __getitem__(self, index):
        sample = self.collection[index]
        target_speaker = sample.speaker[0]
        enroll_index = np.random.choice(self.collection.speaker2audio[target_speaker])

        second_speaker = np.random.choice(list(self.collection.speaker2audio.keys()))
        idx = 0
        while second_speaker == target_speaker and idx < 100:
            second_speaker = np.random.choice(list(self.collection.speaker2audio.keys()))
            idx +=1


        second_speaker_index = np.random.choice(self.collection.speaker2audio[second_speaker])
        target_pt = super().__getitem__(index)['features_list'][0]
        second_pt = super().__getitem__(second_speaker_index)['features_list'][0]
        enroll_pt = super().__getitem__(enroll_index)['features_list'][0]

        print("####1", self.collection[index].audio_file)
        print("####2", self.collection[second_speaker_index].audio_file)
        print("####3", self.collection[enroll_index].audio_file)

        enroll_len = torch.tensor(enroll_pt.shape[0]).long()

        features_list = [target_pt, second_pt]
        features_lengths = [torch.tensor(x.shape[0]).long() for x in features_list]

        min_l = torch.min(torch.stack(features_lengths)).item()

        t1, t2 = [x[:min_l] for x in features_list]

        t1_gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
        t1 = _rescale(t1, t1_gain)
        t2_gain = np.clip(t1_gain + random.normalvariate(-2.51, 2.66), -45, 0)
        t2 = _rescale(t2, t2_gain)
        mix = t1 + t2

        sources = torch.stack([t1, t2], dim=0)
        max_amp = max(torch.abs(mix).max().item(), *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],)

        mix_scaling = 1 / max_amp * 0.9
        t1 = mix_scaling * t1
        t2 = mix_scaling * t2
        mix = mix_scaling * mix
        output = [mix, torch.tensor(min_l).long(), t1, t2, enroll_pt, enroll_len]

        return output

    def _collate_fn(self, batch):
        return _target_audio_to_audio_collate_fn(batch)


def _rescale(x, gain):
    x = x.unsqueeze(0)
    EPS = 1e-14
    avg_amplitude = torch.mean(torch.abs(x), dim=1, keepdim=True)
    normalized = x / (avg_amplitude + EPS)
    out = 10 ** (gain / 20) * normalized
    out = out.squeeze(0)
    return out


class StaticTargetAudioToAudioDataset(_AudioDataset):
    """
    AudioToAudioDataset is intended for Audio to Audio tasks such as speech separation,
    speech enchancement, music source separation etc.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports"""
        output = {}
        output['mixed_audio'] = NeuralType(('B', 'T'), AudioSignal())
        output['mixed_audio_len'] = NeuralType(tuple('B'), LengthsType())
        output['target1'] = NeuralType(('B', 'T'), AudioSignal())
        output['target2'] = NeuralType(('B', 'T'), AudioSignal())
        output['enrollment'] = NeuralType(('B', 'T'), AudioSignal())
        output['enrollment_len'] = NeuralType(tuple('B'), LengthsType())
        return output

    def __init__(
        self,
        manifest_filepath: str,
        featurizer,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            num_sources=3,
            featurizer=featurizer,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            **kwargs,
        )

    def __getitem__(self, index):

        data_pt = super().__getitem__(index)
        features_list = data_pt['features_list']
        enroll_pt = features_list[-1]
        enroll_len = torch.tensor(features_list[-1].shape[0]).long()

        features_list = features_list[:-1]
        scale_factors = data_pt['scale_factors'][:len(features_list)]

        for i, x in enumerate(features_list):
            features_list[i] = _rescale(features_list[i], scale_factors[i])
        features_lengths = [torch.tensor(x.shape[0]).long() for x in features_list]
        min_l = torch.min(torch.stack(features_lengths)).item()
        t1, t2 = [x[:min_l] for x in features_list]
        mix = t1 + t2

        sources = torch.stack([t1, t2], dim=0)
        max_amp = max(torch.abs(mix).max().item(), *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],)

        mix_scaling = 1 / max_amp * 0.9
        t1 = mix_scaling * t1
        t2 = mix_scaling * t2
        mix = mix_scaling * mix

        output = [mix, torch.tensor(min_l).long(), t1, t2, enroll_pt, enroll_len]
        return output

    def _collate_fn(self, batch):
        return _target_audio_to_audio_collate_fn(batch)


def _target_audio_to_audio_collate_fn(batch):
    """collate batch 
    Args:
        batch 
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 6:
        _, audio_lengths, _, _, _, enroll_lengths = packed_batch
    else:
        raise ValueError("Expects 6 tensors in the batch!")

    max_audio_len = 0
    max_enroll_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
        max_enroll_len = max(enroll_lengths).item()

    audio_signal, target1, target2, target3, enroll_signal = [], [], [], [], []
    for b in batch:
        sig, sig_len, t1, t2, enroll_sig, enroll_len = b
        if has_audio:
            sig_len = sig_len.item()
            enroll_len = enroll_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
                t1 = torch.nn.functional.pad(t1, pad)
                t2 = torch.nn.functional.pad(t2, pad)
            if enroll_len < max_enroll_len:
                pad = (0, max_enroll_len - enroll_len)
                enroll_sig = torch.nn.functional.pad(enroll_sig, pad)

            audio_signal.append(sig)
            target1.append(t1)
            target2.append(t2)
            enroll_signal.append(enroll_sig)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        enroll_signal = torch.stack(enroll_signal)
        target1 = torch.stack(target1)
        target2 = torch.stack(target2)
        audio_lengths = torch.stack(audio_lengths)
        enroll_lengths = torch.stack(enroll_lengths)
    else:
        audio_signal, audio_lengths, target1, target2, enroll_signal, enroll_lengths = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    return audio_signal, audio_lengths, target1, target2, enroll_signal, enroll_lengths


def test_AudioToSourceDataset():
    featurizer = WaveformFeaturizer(sample_rate=16000)
    dataset = AudioToSourceDataset(
        manifest_filepath='/media/kpuvvada/data/datasets/data/wsj0-2mix/manifests/test.json',
        num_sources=2,
        featurizer=featurizer,
    )

    item = dataset.__getitem__(1)
    print([x for x in item])


if __name__ == '__main__':
    import nemo

    test_AudioToSourceDataset()
