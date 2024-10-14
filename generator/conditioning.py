import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class ConditioningModule(nn.Module):
    def __init__(self, categorical_dims, embedding_dim, device, alpha=0.9):
        super(ConditioningModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.alpha = alpha

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

        self.mean_embedding = None
        self.cov_embedding = None
        self.inverse_cov_embedding = None  # For Mahalanobis distance
        self.n_samples = 0

    def forward(self, categorical_vars):
        embeddings = []
        for name, embedding in self.category_embeddings.items():
            var = categorical_vars[name].to(self.device)
            embeddings.append(embedding(var))
        conditioning_matrix = torch.cat(embeddings, dim=1)
        conditioning_vector = self.mlp(conditioning_matrix)
        return conditioning_vector

    def initialize_statistics(self, embeddings):
        """
        Initialize mean and covariance using the embeddings from the first batch after warm-up.
        """
        self.mean_embedding = torch.mean(embeddings, dim=0)
        centered_embeddings = embeddings - self.mean_embedding.unsqueeze(0)
        cov_matrix = torch.matmul(centered_embeddings.T, centered_embeddings) / (
            embeddings.size(0) - 1
        )
        cov_matrix += (
            torch.eye(cov_matrix.size(0)).to(self.device) * 1e-6
        )  # For numerical stability
        self.cov_embedding = cov_matrix
        self.inverse_cov_embedding = torch.inverse(self.cov_embedding)
        self.n_samples = embeddings.size(0)


def update_running_statistics(self, embeddings):
    """
    Update mean and covariance using an EWMA algorithm.

    Args:
        embeddings (torch.Tensor): Batch of embedding vectors of shape (batch_size, embedding_dim).
    """
    batch_size = embeddings.size(0)

    if self.n_samples == 0:
        self.initialize_statistics(embeddings)
        return

    # Compute batch mean
    batch_mean = torch.mean(embeddings, dim=0)

    # Update mean using EWMA
    # μ_t = α * μ_{t-1} + (1 - α) * μ_batch
    self.mean_embedding = (
        self.alpha * self.mean_embedding + (1 - self.alpha) * batch_mean
    )

    # Compute batch covariance
    centered_embeddings = embeddings - batch_mean.unsqueeze(0)
    batch_cov = torch.matmul(centered_embeddings.T, centered_embeddings) / (
        batch_size - 1
    )
    batch_cov += (
        torch.eye(batch_cov.size(0)).to(self.device) * 1e-6
    )  # Numerical stability

    # Compute delta (change in mean)
    delta = (
        batch_mean - self.mean_embedding.detach()
    )  # Detach to prevent gradients flowing

    # Update covariance using EWMA
    # Σ_t = α * Σ_{t-1} + (1 - α) * Σ_batch + (1 - α) * δ δ^T
    delta_outer = torch.ger(delta, delta)  # Outer product δ δ^T
    self.cov_embedding = (
        self.alpha * self.cov_embedding
        + (1 - self.alpha) * batch_cov
        + (1 - self.alpha) * delta_outer
    )

    # Update inverse covariance
    cov_embedding_reg = (
        self.cov_embedding
        + torch.eye(self.cov_embedding.size(0)).to(self.device) * 1e-6
    )
    self.inverse_cov_embedding = torch.inverse(cov_embedding_reg)

    # Update effective sample count (optional, based on EWMA)
    # In pure EWMA, sample counts are not tracked, but can be approximated if needed
    # self.n_samples = self.alpha * self.n_samples + (1 - self.alpha) * batch_size

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

    def log_embedding_statistics(
        self,
        epoch,
        writer,
        previous_mean_embedding,
        previous_cov_embedding,
        batch_embeddings,
    ):
        """
        Log embedding statistics to TensorBoard.
        """
        # Log current mean norm and covariance norm

        if previous_mean_embedding is not None:
            mean_embedding_norm = torch.norm(self.mean_embedding).item()
            cov_embedding_norm = torch.norm(self.cov_embedding).item()
        else:
            mean_embedding_norm = 0
            cov_embedding_norm = 0

        writer.add_scalar("Embedding/Mean_Norm", mean_embedding_norm, epoch)
        writer.add_scalar("Embedding/Covariance_Norm", cov_embedding_norm, epoch)

        # Log changes in mean and covariance norms
        if previous_mean_embedding is not None:
            mean_diff = torch.norm(self.mean_embedding - previous_mean_embedding).item()
            writer.add_scalar("Embedding/Mean_Difference", mean_diff, epoch)
        if previous_cov_embedding is not None:
            cov_diff = torch.norm(self.cov_embedding - previous_cov_embedding).item()
            writer.add_scalar("Embedding/Covariance_Difference", cov_diff, epoch)

        # Compute Mahalanobis distances for logging
        sample_embeddings = batch_embeddings
        if sample_embeddings.size(0) > 0:
            mahalanobis_distances = self.compute_mahalanobis_distance(
                sample_embeddings.to(self.device)
            )
            writer.add_histogram(
                "Embedding/Mahalanobis_Distances", mahalanobis_distances.cpu(), epoch
            )

            # Log rarity threshold
            percentile = 0.8  # Same as used in is_rare()
            threshold = torch.quantile(mahalanobis_distances, percentile).item()
            writer.add_scalar("Embedding/Rarity_Threshold", threshold, epoch)
