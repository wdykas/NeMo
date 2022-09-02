import torch
import torch.nn as nn
from collections import OrderedDict
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    MelSpectrogramType,
    Index,
    TokenDurationType
)


"""
Weighted Sum of Pre-trained Speaker Embedding
"""
class Weighted_SpeakerEmbedding(torch.nn.Module):
    def __init__(self, pretrained_embedding):
        super(Weighted_SpeakerEmbedding, self).__init__()
        self.pretrained_embedding = torch.nn.Parameter(pretrained_embedding.weight.detach().clone())
        self.pretrained_embedding.requires_grad = False
        self.num_embeddings = pretrained_embedding.num_embeddings
        self.embedding_weight = torch.nn.Parameter(torch.ones(1, self.num_embeddings))
     
    def forward(self, speaker):
        weight = self.embedding_weight.repeat(len(speaker), 1)
        weight = torch.nn.functional.softmax(weight, dim=-1)
        speaker_emb = weight @ self.pretrained_embedding
        return speaker_emb

    
"""
Global Style Token based Speaker Embedding
"""
class GlobalStyleToken(NeuralModule):
    def __init__(self, 
                 cnn_filters=[32, 32, 64, 64, 128, 128], 
                 dropout=0.2, 
                 gru_hidden=128,
                 gst_size=128, 
                 n_style_token=10, 
                 n_style_attn_head=4):
        super(GlobalStyleToken, self).__init__()    
        self.reference_encoder = ReferenceEncoder_UtteranceLevel(cnn_filters=list(cnn_filters), dropout=dropout, gru_hidden=gru_hidden)
        self.style_attention = StyleAttention(gru_hidden=gru_hidden, gst_size=gst_size, n_style_token=n_style_token, n_style_attn_head=n_style_attn_head)
        
    @property
    def input_types(self):
        return {
            "inp":NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "inp_mask":NeuralType(('B', 'T_spec', 1), TokenDurationType()),
        }

    @property
    def output_types(self):
        return {
            "gst": NeuralType(('B', 'D'), EncodedRepresentation()),
        }
    
    def forward(self, inp, inp_mask):
        style_embedding = self.reference_encoder(inp, inp_mask)
        gst = self.style_attention(style_embedding)
        return gst
    

class ReferenceEncoder_UtteranceLevel(NeuralModule):
    def __init__(self, cnn_filters=[32, 32, 64, 64, 128, 128], dropout=0.2, gru_hidden=128):
        super(ReferenceEncoder_UtteranceLevel, self).__init__()
        self.filter_size = [1] + cnn_filters
        self.dropout = dropout
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    module
                    for i in range(len(cnn_filters))
                    for module in (
                        (
                            "conv2d_{}".format(i + 1),
                            Conv2d(
                                in_channels=int(self.filter_size[i]),
                                out_channels=int(self.filter_size[i + 1]),
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=(1, 1),
                            ),
                        ),
                        ("relu_{}".format(i + 1), nn.ReLU()),
                        (
                            "layer_norm_{}".format(i + 1),
                            nn.LayerNorm(self.filter_size[i + 1]),
                        ),
                        ("dropout_{}".format(i + 1), nn.Dropout(self.dropout)),
                    )
                ]
            )
        )

        self.gru = nn.GRU(
            input_size=cnn_filters[-1] * 2,
            hidden_size=gru_hidden,
            batch_first=True,
        )
        
    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "inputs_masks": NeuralType(('B', 'T_spec', 1), TokenDurationType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'D'), EncodedRepresentation()),
        }
    
    def forward(self, inputs, inputs_masks):
        inputs = inputs.transpose(1,2)
        
        inputs = inputs * inputs_masks
        out = inputs.unsqueeze(3)
        out = self.conv(out)
        out = out.view(out.shape[0], out.shape[1], -1).contiguous()
        self.gru.flatten_parameters()
        memory, out = self.gru(out)
        
        return out.squeeze(0)


class StyleAttention(NeuralModule):
    def __init__(self, gru_hidden=128, gst_size=128, n_style_token=10, n_style_attn_head=4):
        super(StyleAttention, self).__init__()
        self.input_size = gru_hidden
        self.output_size = gst_size
        self.n_token = n_style_token
        self.n_head = n_style_attn_head
        self.token_size = self.output_size // self.n_head

        self.tokens = nn.Parameter(torch.FloatTensor(self.n_token, self.token_size))

        self.q_linear = nn.Linear(self.input_size, self.output_size)
        self.k_linear = nn.Linear(self.token_size, self.output_size)
        self.v_linear = nn.Linear(self.token_size, self.output_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.temperature = (self.output_size // self.n_head) ** 0.5
        nn.init.normal_(self.tokens)
        
    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D'), EncodedRepresentation()),
            "token_id": NeuralType(('B'), Index(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "style_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
        }
    
    def forward(self, inputs, token_id=None):
        bs = inputs.size(0)
        q = self.q_linear(inputs.unsqueeze(1))
        k = self.k_linear(self.tanh(self.tokens).unsqueeze(0).expand(bs, -1, -1))
        v = self.v_linear(self.tanh(self.tokens).unsqueeze(0).expand(bs, -1, -1))

        q = q.view(bs, q.shape[1], self.n_head, self.token_size)
        k = k.view(bs, k.shape[1], self.n_head, self.token_size)
        v = v.view(bs, v.shape[1], self.n_head, self.token_size)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, q.shape[1], q.shape[3])
        k = k.permute(2, 0, 3, 1).contiguous().view(-1, k.shape[3], k.shape[1])
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v.shape[1], v.shape[3])

        scores = torch.bmm(q, k) / self.temperature
        scores = self.softmax(scores)
        if token_id is not None:
            scores = torch.zeros_like(scores)
            scores[:, :, token_id] = 1

        style_emb = torch.bmm(scores, v).squeeze(1)
        style_emb = style_emb.contiguous().view(self.n_head, bs, self.token_size)
        style_emb = style_emb.permute(1, 0, 2).contiguous().view(bs, -1)

        return style_emb
    
class Conv2d(nn.Module):
    """
    Convolution 2D Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 3)
        x = x.contiguous().transpose(2, 3)
        x = self.conv(x)
        x = x.contiguous().transpose(2, 3)
        x = x.contiguous().transpose(1, 3)
        return x