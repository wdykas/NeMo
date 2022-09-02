import torch
import torch.nn as nn
from collections import OrderedDict
from nemo.collections.tts.modules.fastpitch import ConvReLUNorm
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    MelSpectrogramType,
    TokenDurationType,
)



class LayerNorm(nn.Module):
    def __init__(self, nout: int):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(nout, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x.transpose(1, -1))
        x = x.transpose(1, -1)
        return x

class Attentron(NeuralModule):
    def __init__(self, 
        mel_inp_size=80,
        hid_inp_size=384,
        cnn_filters=[512, 512],
        kernel_size=3, 
        stride=1,
        dropout=0.2, 
        gru_hidden=256,
        gru_layer=2,
    ):
        super(Attentron, self).__init__()
        self.filter_size = [mel_inp_size] + list(cnn_filters)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout
        
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    module
                    for i in range(len(cnn_filters))
                    for module in (
                        (
                            "conv1d_{}".format(i + 1),
                            nn.Conv1d(
                                in_channels=int(self.filter_size[i]),
                                out_channels=int(self.filter_size[i + 1]),
                                kernel_size=int(self.kernel_size),
                                stride=int(self.stride),
                                padding=(self.kernel_size - 1) // 2,
                            ),
                        ),
                        (
                            "relu_{}".format(i + 1), 
                            nn.ReLU()
                        ),
                        (
                            "layernorm_{}".format(i + 1),
                            LayerNorm(self.filter_size[i + 1]),
                        ),
                        (
                            "dropout_{}".format(i + 1), 
                            nn.Dropout(self.dropout_rate)
                        ),
                    )
                ]
            )
        )
        self.gru_hidden = gru_hidden
        self.gru_layer = gru_layer
        self.gru = nn.GRU(
            input_size=cnn_filters[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layer,
            batch_first=True,
            bidirectional=True
        )
        
        self.hid_inp_size = hid_inp_size
        self.out_size = hid_inp_size
        self.q_linear = nn.Linear(self.hid_inp_size, self.out_size)
        self.k_linear = nn.Linear(self.gru_hidden * 2, self.out_size)
        self.v_linear = nn.Linear(mel_inp_size, self.out_size)
        
        self.softmax = nn.Softmax(dim=2)
        self.temperature = self.out_size ** 0.5
        
    @property
    def input_types(self):
        return {
            "query": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "ref_spec":NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "emb": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }        
    
    def forward(self, query, ref_spec):
        ref_feat = self.conv(ref_spec)
        
        self.gru.flatten_parameters()
        ref_feat, _ = self.gru(ref_feat.transpose(1,2))
        
        q = self.q_linear(query)
        k = self.k_linear(ref_feat)
        v = self.v_linear(ref_spec.transpose(1,2))
        
        scores = torch.bmm(q, k.transpose(1,2)) / self.temperature
        scores = self.softmax(scores)
        emb = torch.bmm(scores, v)
        return emb
    
    
class Reference_ProsodyEncoder(NeuralModule):
    def __init__(self,
        mel_inp_size=80,
        hid_inp_size=384,
        filter_size=256,
        kernel_size=3, 
        dropout=0.1,
        n_layers=2,       
        gru_hidden=256,
        gru_layer=1,
                 
        use_add_speaker=False, 
        use_cat_speaker=False, 
        use_cln_speaker=False,   
    ):
        super(Reference_ProsodyEncoder, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConvReLUNorm(
                    mel_inp_size + (hid_inp_size if use_cat_speaker else 0) if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout, 
                    speaker_size=hid_inp_size, use_cln_speaker=use_cln_speaker,
                )
            )
                        
        self.gru_hidden = gru_hidden
        self.gru_layer = gru_layer
        self.gru = nn.GRU(
            input_size=filter_size,
            hidden_size=gru_hidden,
            num_layers=gru_layer,
            batch_first=True,
            bidirectional=True
        )
        
        self.hid_inp_size = hid_inp_size
        self.out_size = hid_inp_size
        self.q_linear = nn.Linear(self.hid_inp_size, self.out_size)
        self.k_linear = nn.Linear(self.gru_hidden * 2, self.out_size)
        self.v_linear = nn.Linear(self.gru_hidden * 2, self.out_size)
        
        self.softmax = nn.Softmax(dim=2)
        self.temperature = self.out_size ** 0.5
        
        
        self.use_add_speaker = use_add_speaker
        if use_add_speaker: self.proj_speaker = nn.Linear(self.hid_inp_size, mel_inp_size)
        self.use_cat_speaker = use_cat_speaker

    @property
    def input_types(self):
        return {
            "query": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "query_mask": NeuralType(('B', 'T', 1), TokenDurationType()),
            "ref_spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "ref_spec_mask": NeuralType(('B', 'T_spec', 1), TokenDurationType()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }  
    
    def forward(self, query, query_mask, ref_spec, ref_spec_mask, conditioning=None):
        
        ref_spec = ref_spec.transpose(1,2)
        
        if self.use_add_speaker:
            ref_spec = ref_spec + self.proj_speaker(conditioning)
        
        if self.use_cat_speaker:
            ref_spec = torch.cat([ref_spec, conditioning.repeat(1, ref_spec.shape[1], 1)], dim=-1) 
        
        ref_spec = ref_spec * ref_spec_mask
        ref_feat = ref_spec.transpose(1, 2)
                
        for layer in self.layers:
            ref_feat = layer(ref_feat, conditioning=conditioning)
            
        ref_feat = ref_feat.transpose(1,2) * ref_spec_mask
        self.gru.flatten_parameters()
        ref_feat, _ = self.gru(ref_feat)
        
        q = self.q_linear(query * query_mask) * query_mask
        k = self.k_linear(ref_feat) * ref_spec_mask
        v = self.v_linear(ref_feat) * ref_spec_mask
        
        scores = torch.bmm(q, k.transpose(1,2)) / self.temperature
        scores = self.softmax(scores)
        out = torch.bmm(scores, v)
        out = out * query_mask
        
        return out
    
    
class Target_ProsodyEncoder(NeuralModule):
    def __init__(self,
        mel_inp_size=80,
        hid_inp_size=384,
        filter_size=256,
        kernel_size=3, 
        dropout=0.2, 
        n_layers=2,
        gru_hidden=256,
        gru_layer=1,

        use_avg_before=False,
        use_avg_after=False,
        use_vae=False,
                 
        use_add_speaker=False, 
        use_cat_speaker=False, 
        use_cln_speaker=False, 
    ):
        super(Target_ProsodyEncoder, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConvReLUNorm(
                    mel_inp_size + (hid_inp_size if use_cat_speaker else 0) if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout, 
                    speaker_size=hid_inp_size, use_cln_speaker=use_cln_speaker
                )
            )
        
        
        self.gru_hidden = gru_hidden
        self.gru_layer = gru_layer
        self.gru = nn.GRU(
            input_size=filter_size,
            hidden_size=gru_hidden,
            num_layers=gru_layer,
            batch_first=True,
            bidirectional=True
        )
        
        self.hid_inp_size = hid_inp_size
        self.out_size = hid_inp_size
        
        self.use_vae = use_vae
        if self.use_vae:
            self.vae_mean = nn.Linear(self.gru_hidden * 2, self.out_size)
            self.vae_logvar = nn.Linear(self.gru_hidden * 2, self.out_size)
        else:
            self.proj = nn.Linear(self.gru_hidden * 2, self.out_size)
        
        self.use_avg_before = use_avg_before
        self.use_avg_after  = use_avg_after
        
        
        self.use_add_speaker = use_add_speaker
        if use_add_speaker: self.proj_speaker = nn.Linear(hid_inp_size, mel_inp_size)
        self.use_cat_speaker = use_cat_speaker
        

    @property
    def input_types(self):
        return {
            "tgt_spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "tgt_spec_mask": NeuralType(('B', 'T_spec',  1), TokenDurationType()),
            "durs":NeuralType(('B', 'T_text'), TokenDurationType()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
            "inference": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "prosody_enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "mu" : NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "logvar" : NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }  
    
    def forward(self, tgt_spec, tgt_spec_mask, durs, conditioning=None, inference=False):   

        if self.use_avg_before:
            tgt_spec = self.average_feature(tgt_spec * tgt_spec_mask.transpose(1,2), durs)
            tgt_spec_mask = self.average_feature(tgt_spec_mask.transpose(1,2), durs).transpose(1,2).bool()
            
        tgt_spec = tgt_spec.transpose(1,2)
        if self.use_add_speaker:
            tgt_spec = tgt_spec + self.proj_speaker(conditioning)
        
        if self.use_cat_speaker:
            tgt_spec = torch.cat([tgt_spec, conditioning.repeat(1, tgt_spec.shape[1], 1)], dim=-1) 
        
        tgt_spec = tgt_spec * tgt_spec_mask
        tgt_feat = tgt_spec.transpose(1, 2)
                
        for layer in self.layers:
            tgt_feat = layer(tgt_feat, conditioning=conditioning)
        
        tgt_feat = tgt_feat.transpose(1,2) * tgt_spec_mask
        self.gru.flatten_parameters()
        tgt_feat, _ = self.gru(tgt_feat)
        
        tgt_feat = tgt_feat * tgt_spec_mask
        
        if self.use_avg_after:
            # tgt_feat  = self.average_feature(tgt_feat.transpose(1,2), durs).transpose(1,2)
            tgt_feat  = self.average_gru_feature(tgt_feat.transpose(1,2), durs).transpose(1,2)
            tgt_spec_mask = self.average_feature(tgt_spec_mask.transpose(1,2), durs).transpose(1,2).bool()
            tgt_feat = tgt_feat * tgt_spec_mask
    
        if self.use_vae:
            mu = self.vae_mean(tgt_feat) * tgt_spec_mask
            logvar = self.vae_logvar(tgt_feat) * tgt_spec_mask
            prosody_enc = self.reparameterize(mu, logvar) * tgt_spec_mask if not inference else mu
        else:
            mu, logvar = None, None
            prosody_enc = self.proj(tgt_feat) * tgt_spec_mask
    
        return prosody_enc, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
            
    def average_feature(self, feat, durs):
        """
        feat: (B, D, T_frame)
        durs: (B, T_phoneme)
        """
        durs_cums_ends = torch.cumsum(durs, dim=1).long()
        durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))

        feat_nonzero_cums = torch.nn.functional.pad(torch.cumsum(feat != 0.0, dim=2), (1, 0))
        feat_cums = torch.nn.functional.pad(torch.cumsum(feat, dim=2), (1, 0))

        bs, l = durs_cums_ends.size()
        n_formants = feat.size(1)
        dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
        dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

        feat_sums = (torch.gather(feat_cums, 2, dce) - torch.gather(feat_cums, 2, dcs)).float()
        feat_nelems = (torch.gather(feat_nonzero_cums, 2, dce) - torch.gather(feat_nonzero_cums, 2, dcs)).float()

        feat_avg = torch.where(feat_nelems == 0.0, feat_nelems, feat_sums / feat_nelems)
        return feat_avg
    
    def average_gru_feature(self, feat, durs):
        """
        feat: (B, D, T_frame)
        durs: (B, T_phoneme)
        """
        durs_cums_ends = torch.cumsum(durs, dim=1).long() - 1
        durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1] + 1, (1, 0))
        
        feat_pad  = torch.nn.functional.pad(feat, (0, 1))
        feat_for  = feat_pad[:, :self.gru_hidden, :]
        feat_back = feat_pad[:, self.gru_hidden:, :]
        
        bs, l = durs_cums_ends.size()
        n_formants = self.gru_hidden
        dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
        dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)
        
        feat_for = torch.gather(feat_for, 2, dce)
        feat_back = torch.gather(feat_back, 2, dcs)
        feat_avg = torch.cat((feat_for, feat_back), dim=1)
        return feat_avg
        
class Target_ProsodyPredictor(NeuralModule):
    def __init__(self,
        hid_inp_size=384,
        filter_size=256,
        kernel_size=3, 
        dropout=0.2, 
        n_layers=2,
        gru_hidden=256,
        gru_layer=1,

        use_add_speaker=False, 
        use_cat_speaker=False, 
        use_cln_speaker=False, 
    ):
        super(Target_ProsodyPredictor, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConvReLUNorm(
                    hid_inp_size + (hid_inp_size if use_cat_speaker else 0) if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout, 
                    speaker_size=hid_inp_size, use_cln_speaker=use_cln_speaker,
                )
            )
        
        
        self.gru_hidden = gru_hidden
        self.gru_layer = gru_layer
        self.gru = nn.GRU(
            input_size=filter_size,
            hidden_size=gru_hidden,
            num_layers=gru_layer,
            batch_first=True,
            bidirectional=True
        )
        
        self.hid_inp_size = hid_inp_size
        self.out_size = hid_inp_size
        self.proj = nn.Linear(self.gru_hidden * 2, self.out_size)
                
        self.use_add_speaker = use_add_speaker
        self.use_cat_speaker = use_cat_speaker
    
    @property
    def input_types(self):
        return {
            "enc_out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "enc_mask": NeuralType(('B', 'T', 1), TokenDurationType()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "prosody_prd": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }  
        
    def forward(self, enc_out, enc_mask, conditioning=None):    

        
        if self.use_add_speaker:
            enc_out = enc_out + conditioning
        
        if self.use_cat_speaker:
            enc_out = torch.cat([enc_out, conditioning.repeat(1, enc_out.shape[1], 1)], dim=-1) 
        
        out = enc_out * enc_mask
        out = out.transpose(1, 2)
        
        for layer in self.layers:
            out = layer(out, conditioning=conditioning)
            
        out = out.transpose(1,2) * enc_mask
        
        self.gru.flatten_parameters()
        out, _ = self.gru(out)
        prosody_prd = self.proj(out) * enc_mask 
            
        return prosody_prd