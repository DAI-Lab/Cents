import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditioningModule(nn.Module):
    def __init__(self, categorical_dims, numerical_dims, embedding_dim, device):
        super(ConditioningModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.category_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_categories + 1, embedding_dim).to(device)  # +1 for 'missing' token
            for name, num_categories in categorical_dims.items()
        })
        self.numerical_projections = nn.ModuleDict({
            name: nn.Linear(1, embedding_dim).to(device)
            for name in numerical_dims
        })
        total_dim = (len(categorical_dims) + len(numerical_dims)) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        ).to(device)

    def forward(self, categorical_vars, numerical_vars):
        embeddings = []
        for name, embedding in self.category_embeddings.items():
            var = categorical_vars[name].to(self.device)
            embeddings.append(embedding(var))
        for name, projection in self.numerical_projections.items():
            var = numerical_vars[name].unsqueeze(1).to(self.device)  # Ensure shape (batch_size, 1)
            embeddings.append(projection(var))
        conditioning_vector = torch.cat(embeddings, dim=1)
        conditioning_vector = self.mlp(conditioning_vector)
        return conditioning_vector
