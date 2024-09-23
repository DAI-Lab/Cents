import torch
import torch.nn as nn


class ConditioningModule(nn.Module):
    def __init__(self, categorical_dims, embedding_dim, device):
        super(ConditioningModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.category_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_categories, embedding_dim).to(device)
                for name, num_categories in categorical_dims.items()
            }
        )
        total_dim = len(categorical_dims) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            # nn.Linear(total_dim, embedding_dim)
        ).to(device)

    def forward(self, categorical_vars):
        embeddings = []
        for name, embedding in self.category_embeddings.items():
            var = categorical_vars[name].to(self.device)
            embeddings.append(embedding(var))
        conditioning_matrix = torch.cat(embeddings, dim=1)
        conditioning_matrix = self.mlp(conditioning_matrix)
        return conditioning_matrix
