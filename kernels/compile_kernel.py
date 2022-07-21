import fused_kernels
import torch
fused_kernels.load()


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def forward_torch_softmax(input, mask, scale):
    input = input * scale
    mask_output = attention_mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    return probs


import fused_kernels.build.scaled_masked_softmax_cuda
print(fused_kernels.build.scaled_masked_softmax_cuda.forward)

scale_t = torch.tensor([1.0])

inputs = torch.rand((2, 4, 32, 32), dtype=torch.float16, device='cuda:0')
masks =  torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.bool, device='cuda:0')
# inputs = torch.rand((1, 1, 1, 4096), dtype=torch.float16, device='cuda:0')
# masks =  torch.randint(0, 2, (1, 1, 1, 4096), dtype=torch.bool, device='cuda:0')


softmax_results = fused_kernels.build.scaled_masked_softmax_cuda.forward(inputs, masks, scale_t[0])
softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())

print((softmax_results_torch - softmax_results).abs().max())
print(softmax_results)