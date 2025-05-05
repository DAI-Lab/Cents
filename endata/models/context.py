import torch
import torch.nn as nn


class ContextModule(nn.Module):
    """
    Integrates multiple context variables into a single embedding and provides
    auxiliary classification logits for each variable.

    This module:
        - Learns separate embeddings for each context variable.
        - Concatenates embeddings and projects through an MLP to a shared embedding.
        - Outputs classification logits per context variable for auxiliary loss.

    Attributes:
        context_embeddings (nn.ModuleDict): Embedding layers for each variable.
        mlp (nn.Sequential): MLP to combine embeddings into a single vector.
        classification_heads (nn.ModuleDict): Linear heads for per-variable logits.
    """

    def __init__(self, context_vars: dict[str, int], embedding_dim: int):
        """
        Initialize the ContextModule.

        Args:
            context_vars (Dict[str, int]): Mapping of variable names to category counts.
            embedding_dim (int): Size of each variable's embedding vector.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        self.context_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_categories, embedding_dim)
                for name, num_categories in context_vars.items()
            }
        )

        total_dim = len(context_vars) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        self.classification_heads = nn.ModuleDict(
            {
                var_name: nn.Linear(embedding_dim, num_categories)
                for var_name, num_categories in context_vars.items()
            }
        )

    def forward(
        self, context_vars: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute a combined context embedding and classification logits.

        Args:
            context_vars (Dict[str, Tensor]): Tensors of category indices per variable.

        Returns:
            embedding (Tensor): Combined embedding of shape (batch_size, embedding_dim).
            classification_logits (Dict[str, Tensor]): Logits per variable,
                each of shape (batch_size, num_categories).
        """
        embeddings = [
            layer(context_vars[name]) for name, layer in self.context_embeddings.items()
        ]

        context_matrix = torch.cat(embeddings, dim=1)
        embedding = self.mlp(context_matrix)

        classification_logits = {
            var_name: head(embedding)
            for var_name, head in self.classification_heads.items()
        }

        return embedding, classification_logits
