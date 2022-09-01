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