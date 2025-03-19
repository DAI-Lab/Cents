import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextModule(nn.Module):
    def __init__(self, context_vars, embedding_dim, device):
        """
        A context module that:
          - Learns separate embeddings for each context variable.
          - Concatenates them and passes through an MLP to produce a single
            fixed-size embedding.
          - Provides classification logits for each context variable to compute
            an auxiliary classification loss.

        Args:
            context_vars (dict): {var_name: num_categories} for each context variable.
            embedding_dim (int): Dimension of the shared latent embedding for each context variable.
            device: Torch device (CPU or GPU).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.context_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_categories, embedding_dim)
                for name, num_categories in context_vars.items()
            }
        ).to(device)

        total_dim = len(context_vars) * embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        ).to(device)

        self.classification_heads = nn.ModuleDict()
        for var_name, num_categories in context_vars.items():
            self.classification_heads[var_name] = nn.Linear(
                embedding_dim, num_categories
            ).to(device)

    def forward(self, context_vars):
        """
        Forward pass to compute:
          - A single embedding that integrates all context vars
          - A dict of classification logits (one head per context var)

        Args:
            context_vars (dict): e.g. {var_name: Tensor[int64]}

        Returns:
            embedding (Tensor): shape (batch_size, embedding_dim)
            classification_logits (dict): {var_name: Tensor of shape (batch_size, num_categories)}
        """
        embeddings = []
        for name, embedding_layer in self.context_embeddings.items():
            cat_tensor = context_vars[name].to(self.device)
            embeddings.append(embedding_layer(cat_tensor))

        context_matrix = torch.cat(embeddings, dim=1)
        embedding = self.mlp(context_matrix)

        classification_logits = {}
        for var_name, head in self.classification_heads.items():
            classification_logits[var_name] = head(embedding)

        return embedding, classification_logits
