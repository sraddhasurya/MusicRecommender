import torch.nn as nn
import torch 
import torch.nn.functional as F

class SimilarityModel(nn.Module,):
    def __init__(self, total_feature_dim):
        super(SimilarityModel, self).__init__()
        self.feature_weights=nn.Parameter(torch.randn(total_feature_dim))

    def forward(self, x1,x2):
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)

        # Apply learned weights
        weighted_1 = x1 * self.feature_weights
        weighted_2 = x2 * self.feature_weights

        # Compute cosine similarity
        similarity = F.cosine_similarity(weighted_1, weighted_2, dim=-1)
        return similarity