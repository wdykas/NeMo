import torch

class Weighted_SpeakerEmbedding(torch.nn.Module):
    def __init__(self, pretrained_embedding):
        super(SpeakerEmbedding, self).__init__()
        self.pretrained_embedding = torch.nn.Parameter(pretrained_embedding.weight.detach().clone())
        self.pretrained_embedding.requires_grad = False
        self.num_embeddings = pretrained_embedding.num_embeddings
        self.embedding_weight = torch.nn.Parameter(torch.ones(1, self.num_embeddings))
     
    def forward(self, speaker):
        weight = self.embedding_weight.repeat(len(speaker), 1)
        weight = torch.nn.functional.softmax(weight, dim=-1)
        speaker_emb = weight @ self.pretrained_embedding
        return speaker_emb