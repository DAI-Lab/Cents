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
        ).to(device)

        # Variables for collecting embeddings and computing Gaussian parameters
        self.embeddings_list = []
        self.mean_embedding = None
        self.cov_embedding = None
        self.inverse_cov_embedding = None  # For Mahalanobis distance

    def forward(self, categorical_vars):
        embeddings = []
        for name, embedding in self.category_embeddings.items():
            var = categorical_vars[name].to(self.device)
            embeddings.append(embedding(var))
        conditioning_matrix = torch.cat(embeddings, dim=1)
        conditioning_vector = self.mlp(conditioning_matrix)
        return conditioning_vector

    def collect_embeddings(self, categorical_vars):
        """
        Collect conditional embeddings during the warm-up period.
        """
        with torch.no_grad():
            embedding = self.forward(categorical_vars)
            self.embeddings_list.append(embedding.cpu())

    def compute_gaussian_parameters(self):
        """
        Compute mean and covariance of the collected embeddings.
        """
        all_embeddings = torch.cat(self.embeddings_list, dim=0)
        self.mean_embedding = torch.mean(all_embeddings, dim=0).to(self.device)
        # Compute covariance
        centered_embeddings = all_embeddings - self.mean_embedding.cpu()
        cov_matrix = torch.matmul(centered_embeddings.T, centered_embeddings) / (
            all_embeddings.size(0) - 1
        )
        cov_matrix += torch.eye(cov_matrix.size(0)) * 1e-6
        self.cov_embedding = cov_matrix.to(self.device)
        self.inverse_cov_embedding = torch.inverse(self.cov_embedding)

    def compute_mahalanobis_distance(self, embeddings):
        """
        Compute Mahalanobis distance for the given embeddings.
        """
        diff = embeddings - self.mean_embedding.unsqueeze(0)
        left = torch.matmul(diff, self.inverse_cov_embedding)
        mahalanobis_distance = torch.sqrt(torch.sum(left * diff, dim=1))
        return mahalanobis_distance

    def is_rare(self, embeddings, threshold=None, percentile=0.8):
        """
        Determine if the embeddings are rare based on Mahalanobis distance.

        Args:
            embeddings (torch.Tensor): The embeddings to evaluate.
            threshold (float, optional): Specific Mahalanobis distance threshold.
            percentile (float, optional): Percentile to define rarity if threshold is not provided.

        Returns:
            rare_mask (torch.Tensor): A boolean mask indicating rare embeddings.
        """
        mahalanobis_distance = self.compute_mahalanobis_distance(embeddings)
        if threshold is None:
            threshold = torch.quantile(mahalanobis_distance, percentile)
        rare_mask = mahalanobis_distance > threshold
        return rare_mask
