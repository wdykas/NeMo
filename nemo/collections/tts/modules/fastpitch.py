# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#
# BSD 3-Clause License
#
# Copyright (c) 2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from typing import Optional, List
from omegaconf import DictConfig, OmegaConf

import torch
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.tts.helpers.helpers import binarize_attention_parallel, regulate_len
from nemo.core.classes import NeuralModule, typecheck, adapter_mixins
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    Index,
    LengthsType,
    LogprobsType,
    MelSpectrogramType,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.collections.tts.modules.transformer import ConditionalLayerNorm

def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg


class ConvReLUNorm(torch.nn.Module, adapter_mixins.AdapterModuleMixin):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0, speaker_size=384, use_cln_speaker=False):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.use_cln_speaker = use_cln_speaker
        
        if use_cln_speaker:
            self.norm = ConditionalLayerNorm(out_channels, spk_emb_dim=speaker_size)
        else:
            self.norm = torch.nn.LayerNorm(out_channels)
        
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal, conditioning=None):
        
        out = torch.nn.functional.relu(self.conv(signal))
        
        if self.use_cln_speaker and conditioning is not None:
            out = self.norm(out.transpose(1, 2), conditioning).transpose(1, 2)
        else:
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
            
        out = self.dropout(out)
        
        if self.is_adapter_available():
            out = self.forward_enabled_adapters(out.transpose(1,2)).transpose(1, 2)
            
        return out


class TemporalPredictor(NeuralModule):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, use_add_speaker=False, use_cat_speaker=False, use_cln_speaker=False, n_layers=2):
        super(TemporalPredictor, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConvReLUNorm(
                    input_size * (2 if use_cat_speaker else 1) if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout, 
                    speaker_size=input_size, use_cln_speaker=use_cln_speaker, 
                )
            )
        self.fc = torch.nn.Linear(filter_size, 1, bias=True)
        self.use_add_speaker = use_add_speaker
        self.use_cat_speaker = use_cat_speaker
        self.filter_size = filter_size

    @property
    def input_types(self):
        return {
            "enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "enc_mask": NeuralType(('B', 'T', 1), TokenDurationType()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T'), EncodedRepresentation()),
        }

    def forward(self, enc, enc_mask, conditioning=None):
        if self.use_add_speaker:
            enc = enc + conditioning
        
        if self.use_cat_speaker:
            enc = torch.cat([enc, conditioning.repeat(1, enc.shape[1], 1)], dim=-1) 
        
        out = enc * enc_mask
        out = out.transpose(1, 2)
        
        for layer in self.layers:
            out = layer(out, conditioning=conditioning)
            
        out = out.transpose(1, 2)
        out = self.fc(out) * enc_mask
        return out.squeeze(-1)
    
    
class TemporalPredictorAdapter(TemporalPredictor, adapter_mixins.AdapterModuleMixin):
    
    # Higher level forwarding
    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for conv_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conv_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([conv_layer.is_adapter_available() for conv_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for conv_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conv_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for conv_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            names.update(conv_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.filter_size)
        return cfg


"""
Register any additional information
"""
if adapter_mixins.get_registered_adapter(TemporalPredictor) is None:
    adapter_mixins.register_adapter(base_class=TemporalPredictor, adapter_class=TemporalPredictorAdapter)

class FastPitchModule(NeuralModule):
    def __init__(
        self,
        encoder_module: NeuralModule,
        decoder_module: NeuralModule,
        duration_predictor: NeuralModule,
        pitch_predictor: NeuralModule,
        aligner: NeuralModule,
        
        gst_model: NeuralModule,
        sv_model: str,
        attentron_model: NeuralModule,
        reference_prosodyencoder: NeuralModule,
        target_prosodyencoder: NeuralModule,
        target_prosodypredictor: NeuralModule,
        
        n_speakers: int,
        symbols_embedding_dim: int,
        pitch_embedding_kernel_size: int,
        n_mel_channels: int = 80,
        max_token_duration: int = 75,
        sv_sample_rate: int = 16000,
        
        use_lookup_speaker: bool = True,
        use_gst_speaker: bool = False,
        use_sv_speaker: bool = False,
        
        use_attentron: bool = False,
        use_reference_prosodyencoder: bool = False,
        use_target_prosodyencoder: bool = False,
    ):
        super().__init__()

        self.encoder = encoder_module
        self.decoder = decoder_module
        self.duration_predictor = duration_predictor
        self.pitch_predictor = pitch_predictor
        self.aligner = aligner
        self.learn_alignment = aligner is not None
        self.use_duration_predictor = True
        self.binarize = False

        self.speaker_emb = None
        self.gst_speaker_emb = None
        if n_speakers > 1:
            if use_lookup_speaker: self.speaker_emb = torch.nn.Embedding(n_speakers, symbols_embedding_dim)
            if use_gst_speaker: self.gst_speaker_emb = gst_model
            if use_sv_speaker: 
                sv_model = EncDecSpeakerLabelModel.from_pretrained(model_name=sv_model)
                config = OmegaConf.create(dict(manifest_filepath=None, sample_rate=sv_sample_rate, labels=None))
                sv_model.setup_test_data(config)
                self.sv_speaker_emb = sv_model
                self.sv_linear = torch.nn.Linear(sv_model.cfg.decoder.emb_sizes, symbols_embedding_dim)
        
        self.attentron_model = None
        if use_attentron:
            self.attentron_model = attentron_model
        
        self.reference_prosodyencoder = None
        if use_reference_prosodyencoder:
            self.reference_prosodyencoder = reference_prosodyencoder
            
        self.target_prosodyencoder = None
        self.target_prosodypredictor = None
        if use_target_prosodyencoder:
            self.target_prosodyencoder = target_prosodyencoder
            self.target_prosodypredictor = target_prosodypredictor
            
        self.max_token_duration = max_token_duration
        self.min_token_duration = 0

        self.pitch_emb = torch.nn.Conv1d(
            1,
            symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size,
            padding=int((pitch_embedding_kernel_size - 1) / 2),
        )

        # Store values precomputed from training data for convenience
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.proj = torch.nn.Linear(self.decoder.d_model * 2 if use_attentron else self.decoder.d_model, n_mel_channels, bias=True)
        
    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "speaker": NeuralType(('B'), Index(), optional=True),
            "pace": NeuralType(optional=True),
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "ref_spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "ref_spec_lens": NeuralType(('B'), LengthsType(), optional=True),
            "ref_audio": NeuralType(('B', 'T_audio'), RegressionValuesType(), optional=True),
            "ref_audio_lens": NeuralType(('B'), LengthsType(), optional=True),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
            "mel_lens": NeuralType(('B'), LengthsType(), optional=True),
            "input_lens": NeuralType(('B'), LengthsType(), optional=True),
            "learn_prosody_predictor": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "num_frames": NeuralType(('B'), TokenDurationType()),
            "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
            "log_durs_predicted": NeuralType(('B', 'T_text'), TokenLogDurationType()),
            "pitch_predicted": NeuralType(('B', 'T_text'), RegressionValuesType()),
            "attn_soft": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "attn_logprob": NeuralType(('B', 'S', 'T_spec', 'T_text'), LogprobsType()),
            "attn_hard": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "attn_hard_dur": NeuralType(('B', 'T_text'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "prosody_predict": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
            "prosody_encode": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
            "mu": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "logvar": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        speaker=None,
        pace=1.0,
        spec=None,
        ref_spec=None,
        ref_spec_lens=None,
        ref_audio=None,
        ref_audio_lens=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
        learn_prosody_predictor=None,
    ):

        if not self.learn_alignment and self.training:
            assert durs is not None
            assert pitch is not None
        
        # Calculate speaker embedding
        spk_emb = 0
        
        # Lookup Speaker Embedding
        if self.speaker_emb is not None and speaker is not None:
            assert speaker.max() < self.speaker_emb.num_embeddings
            speaker[(speaker < 0)] += self.speaker_emb.num_embeddings
            spk_emb += self.speaker_emb(speaker).unsqueeze(1)
        
        # GST Speaker Embedding
        if self.gst_speaker_emb is not None and ref_spec is not None and ref_spec_lens is not None:     
            ref_spec_mask = (torch.arange(ref_spec_lens.max()).to(ref_spec.device).expand(ref_spec_lens.shape[0], ref_spec_lens.max()) < ref_spec_lens.unsqueeze(1)).unsqueeze(2)
            spk_emb += self.gst_speaker_emb(ref_spec, ref_spec_mask).unsqueeze(1)
        
        # SV Speaker Embedding
        if self.sv_speaker_emb is not None and ref_audio is not None and ref_audio_lens is not None:
            with torch.no_grad():
                _, embs = self.sv_speaker_emb(input_signal=ref_audio, input_signal_length=ref_audio_lens)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            embs = self.sv_linear(embs)
            spk_emb += embs.unsqueeze(1)
            
        # Input FFT
        enc_out, enc_mask = self.encoder(input=text, conditioning=spk_emb)
        
        # Alignment
        attn_soft, attn_hard, attn_hard_dur, attn_logprob = None, None, None, None
        if self.learn_alignment and spec is not None:
            text_emb = self.encoder.word_emb(text)
            attn_soft, attn_logprob = self.aligner(spec, text_emb.permute(0, 2, 1), enc_mask == 0, attn_prior, conditioning=spk_emb)
            attn_hard = binarize_attention_parallel(attn_soft, input_lens, mel_lens)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        
        # Add Reference Prosody
        if self.reference_prosodyencoder is not None and ref_spec is not None and ref_spec_lens is not None:
            # TODO [ref_spec_lens MASK]
            ref_spec_mask = (torch.arange(ref_spec_lens.max()).to(ref_spec.device).expand(ref_spec_lens.shape[0], ref_spec_lens.max()) < ref_spec_lens.unsqueeze(1)).unsqueeze(2)
            prosody_ref = self.reference_prosodyencoder(enc_out, enc_mask, ref_spec, ref_spec_mask, conditioning=spk_emb)
            enc_out = enc_out + prosody_ref
        
        # Add Target Encode/Predict prosody 
        prosody_encode, prosody_predict, mu, logvar = None, None, None, None
        if self.target_prosodyencoder is not None and self.target_prosodypredictor is not None:
            if spec is not None and mel_lens is not None:
                spec_mask = (torch.arange(mel_lens.max()).to(spec.device).expand(mel_lens.shape[0], mel_lens.max()) < mel_lens.unsqueeze(1)).unsqueeze(2)
                if learn_prosody_predictor:
                    with torch.no_grad():
                        prosody_encode, _, _ = self.target_prosodyencoder(spec, spec_mask, durs=attn_hard_dur, conditioning=spk_emb, inference=True) 
                    prosody_encode = (prosody_encode * enc_mask).detach()
                    prosody_predict = self.target_prosodypredictor(enc_out, enc_mask, conditioning=spk_emb)
                else:
                    prosody_encode, mu, logvar = self.target_prosodyencoder(spec, spec_mask, durs=attn_hard_dur, conditioning=spk_emb)
                    prosody_encode = (prosody_encode * enc_mask)
                    
                prosody_tgt = prosody_encode
            else:
                prosody_predict = self.target_prosodypredictor(enc_out, enc_mask, conditioning=spk_emb)
                prosody_tgt = prosody_predict
                
            enc_out = enc_out + prosody_tgt
            
        # Predict duration
        log_durs_predicted = self.duration_predictor(enc_out, enc_mask, conditioning=spk_emb)
        durs_predicted = torch.clamp(torch.exp(log_durs_predicted) - 1, 0, self.max_token_duration)

        # Predict pitch
        pitch_predicted = self.pitch_predictor(enc_out, enc_mask, conditioning=spk_emb)
        if pitch is not None:
            if self.learn_alignment and pitch.shape[-1] != pitch_predicted.shape[-1]:
                # Pitch during training is per spectrogram frame, but during inference, it should be per character
                pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if self.learn_alignment and spec is not None:
            len_regulated, dec_lens = regulate_len(attn_hard_dur, enc_out, pace)
        elif spec is None and durs is not None:
            len_regulated, dec_lens = regulate_len(durs, enc_out, pace)
        # Use predictions during inference
        elif spec is None:
            len_regulated, dec_lens = regulate_len(durs_predicted, enc_out, pace)

        
        # Output FFT
        dec_out, _ = self.decoder(input=len_regulated, seq_lens=dec_lens, conditioning=spk_emb)
        
        # [Prosody] Attentron
        if self.attentron_model is not None and ref_spec is not None:
            attentron_out = self.attentron_model(dec_out, ref_spec)
            dec_out = torch.cat((dec_out, attentron_out), dim=-1)
        
        spect = self.proj(dec_out).transpose(1, 2)
        return (
            spect,
            dec_lens,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            pitch,
            prosody_predict,
            prosody_encode,
            mu,
            logvar,
        )

    def infer(self, *, text, pitch=None, speaker=None, pace=1.0, volume=None):
        # Calculate speaker embedding
        spk_emb = 0
        
        # Lookup Speaker Embedding
        if self.speaker_emb is not None and speaker is not None:
            assert speaker.max() < self.speaker_emb.num_embeddings
            speaker[(speaker < 0)] += self.speaker_emb.num_embeddings
            spk_emb += self.speaker_emb(speaker).unsqueeze(1)
        
        # GST Speaker Embedding
        if self.gst_speaker_emb is not None and ref_spec is not None and ref_spec_lens is not None:     
            ref_spec_mask = (torch.arange(ref_spec_lens.max()).to(ref_spec.device).expand(ref_spec_lens.shape[0], ref_spec_lens.max()) < ref_spec_lens.unsqueeze(1)).unsqueeze(2)
            spk_emb += self.gst_speaker_emb(ref_spec, ref_spec_mask).unsqueeze(1)

        # SV Speaker Embedding
        if self.sv_speaker_emb is not None and ref_audio is not None and ref_audio_lens is not None:
            with torch.no_grad():
                _, embs = self.sv_speaker_emb(input_signal=ref_audio, input_signal_length=ref_audio_lens)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            embs = self.sv_linear(embs)
            spk_emb += embs.unsqueeze(1)
            
        # Input FFT
        enc_out, enc_mask = self.encoder(input=text, conditioning=spk_emb)
        
        # Add Reference Prosody
        if self.reference_prosodyencoder is not None and ref_spec is not None and ref_spec_lens is not None:
            ref_spec_mask = (torch.arange(ref_spec_lens.max()).to(ref_spec.device).expand(ref_spec_lens.shape[0], ref_spec_lens.max()) < ref_spec_lens.unsqueeze(1)).unsqueeze(2)
            prosody_ref = self.reference_prosodyencoder(enc_out, enc_mask, ref_spec, ref_spec_mask, conditioning=spk_emb)
            enc_out = enc_out + prosody_ref
            
        # Add Target Predict Prosody 
        prosody_encode, prosody_predict = None, None
        if self.target_prosodypredictor is not None:
            prosody_tgt = self.target_prosodypredictor(enc_out, enc_mask, conditioning=spk_emb)
            enc_out += prosody_tgt
        
        # Predict duration and pitch
        log_durs_predicted = self.duration_predictor(enc_out, enc_mask, conditioning=spk_emb)
        durs_predicted = torch.clamp(
            torch.exp(log_durs_predicted) - 1.0, self.min_token_duration, self.max_token_duration
        )
        pitch_predicted = self.pitch_predictor(enc_out, enc_mask, conditioning=spk_emb) + pitch
        pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        # Expand to decoder time dimension
        len_regulated, dec_lens = regulate_len(durs_predicted, enc_out, pace)
        volume_extended = None
        if volume is not None:
            volume_extended, _ = regulate_len(durs_predicted, volume.unsqueeze(-1), pace)
            volume_extended = volume_extended.squeeze(-1).float()

        # Output FFT
        dec_out, _ = self.decoder(input=len_regulated, seq_lens=dec_lens, conditioning=spk_emb)
        
        # [Prosody] Attentron
        if self.attentron is not None and ref_spec is not None:
            attentron_out = self.attentron_model(dec_out, ref_spec)
            dec_out = torch.cat((dec_out, attentron_out), dim=-1)
        
        spect = self.proj(dec_out).transpose(1, 2)
        return (
            spect.to(torch.float),
            dec_lens,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            volume_extended,
            prosody_predict,
            prosody_encode,
        )

    
