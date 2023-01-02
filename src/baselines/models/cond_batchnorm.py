import torch
import torch.nn as nn

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        
    def forward(self, x, y):
        # x.shape == (N,C,L)
        # y.shape == (N,classes)
        
        batch_size = x.size(0)
        out = self.bn(x)
        emb = y @ self.embed.weight
        gamma, beta = emb.chunk(2,1)
        gamma_reshaped = gamma.reshape(batch_size, self.num_features, -1)
        beta_reshaped = beta.reshape(batch_size, self.num_features, -1)
        out = gamma_reshaped * out + beta_reshaped
        return out
      
