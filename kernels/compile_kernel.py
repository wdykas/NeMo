import torch
import time
import scaled_masked_softmax_cuda
import nemo.collections.nlp.modules.common.megatron.fused_kernels 
import scaled_masked_softmax_cuda_new



def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def forward_torch_softmax(input, mask, scale):
    input = input * scale
    mask_output = attention_mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    return probs


print(scaled_masked_softmax_cuda_new.forward)

scale_t = torch.tensor([1.0])

# inputs = torch.rand((2, 4, 323, 3222), dtype=torch.float16, device='cuda:0')
# masks =  torch.randint(0, 2, (1, 1, 323, 3222), dtype=torch.bool, device='cuda:0')
# masks =  torch.zeros((1, 1, 323, 3222), dtype=torch.bool, device='cuda:0')
batch = 2 
attn = 16
qlen = 2348
klen = 3123
# inputs = torch.rand((batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0')
# masks =  torch.randint(0, 2, (batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
inputs = torch.load('/results/input.pth').contiguous().to('cuda:0')
masks = torch.load('/results/mask.pth').contiguous().to('cuda:0')
#scale_t = torch.tensor(torch.load('/results/scale.pth'), dtype=torch.float)
# inputs = torch.rand((1, 1, 2, 32), dtype=torch.float16, device='cuda:0')
# masks =  torch.randint(0, 2, (1, 1, 2, 32), dtype=torch.bool, device='cuda:0')
# backward = torch.rand((2, 4, 323, 3222), dtype=torch.float16, device='cuda:0')
backward = torch.rand_like(inputs, dtype=torch.float16, device='cuda:0')
backward2 = backward.clone()
# i=0
# j=2
# k=18
# # max_v = inputs[i:i+1, j:j+1, k:k+1,:].max()
# # 
# ins = inputs[i:i+1, j:j+1, k:k+1,:].contiguous()
# mas = masks[i:i+1, 0:1, k:k+1,:].contiguous()
# print(max_v)
# scaled_masked_softmax_cuda_new.forward(ins, mas, scale_t[0])
# 
# for i in range(256):
#     for j in range(12):
#         for k in range(64):
#             print(i,j,k, scaled_masked_softmax_cuda_new.forward(inputs[i:i+1, j:j+1, k:k+1,:], masks[i:i+1, j:j+1, k:k+1, :], scale_t[0]).mean())

softmax_results = scaled_masked_softmax_cuda_new.forward(inputs, masks, scale_t[0].item())

# back_grad = scaled_masked_softmax_cuda_new.backward(backward, softmax_results, scale_t[0])

#inputs.requires_grad = True
softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())
softmax_results_torch.backward(backward2)
print((softmax_results_torch - softmax_results).abs().max())
print((back_grad - inputs.grad).abs().max())

# print(softmax_results)
# forward mode
torch.autograd.functional.jvp(lambda i: forward_torch_softmax(i, masks, scale_t[0]), inputs, inputs)


beg = time.time()
for i in range(30):
    scaled_masked_softmax_cuda.forward(inputs, masks, scale_t[0])
torch.cuda.synchronize(device='cuda:0')
end = time.time()
print('apex',end - beg)


beg = time.time()
for i in range(30):
    softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())
torch.cuda.synchronize(device='cuda:0')
end = time.time()
print('torch',end - beg)

beg = time.time()
for i in range(30):
    scaled_masked_softmax_cuda_new.forward(inputs, masks, scale_t[0])
torch.cuda.synchronize(device='cuda:0')
end = time.time()
print('yi',end - beg)
