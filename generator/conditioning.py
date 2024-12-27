import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


class ConditioningModule(nn.Module):
    def __init__(self, categorical_dims, embedding_dim, device, n_components=10):
        """
        Args:
            categorical_dims (dict): {var_name: num_categories} for each conditioning variable.
            embedding_dim (int): Dimension of the latent embedding for each categorical variable.
            device: Torch device (CPU or GPU).
            n_components (int): Number of components for the Gaussian Mixture Model.
            kl_alpha (float): Weight for KL regularization (hyperparam).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.category_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_categories, embedding_dim)
                for name, num_categories in categorical_dims.items()
            }
        ).to(device)

        total_dim = len(categorical_dims) * embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * embedding_dim),
        ).to(device)

        # ====== GMM-Related Attributes ======
        self.n_components = n_components
        self.gmm = None  # Will hold a fitted GaussianMixture instance
        self.log_prob_threshold = None  # Used to define rarity

    def forward(self, categorical_vars, sample=True):
        """
        Forward pass to compute mu and (optionally) sample z via reparameterization.

        Args:
            categorical_vars (dict): e.g. {var_name: Tensor[int64]}
            sample (bool): If True, return z = mu + sigma*eps, else return mu.

        Returns:
            z (Tensor): The sampled embedding or just mu (shape: [batch_size, embedding_dim]).
            mu (Tensor): The mean of the embedding distribution (shape: [batch_size, embedding_dim]).
            logvar (Tensor): Log-variance (shape: [batch_size, embedding_dim]).
        """
        embeddings = []
        for name, embedding_layer in self.category_embeddings.items():
            cat_tensor = categorical_vars[name].to(self.device)
            embeddings.append(embedding_layer(cat_tensor))

        conditioning_matrix = torch.cat(embeddings, dim=1)
        stats = self.mlp(conditioning_matrix)
        mu = stats[:, : self.embedding_dim]
        logvar = stats[:, self.embedding_dim :]

        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        return z, mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Standard reparameterization trick: z = mu + sigma * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        """
        KL( q(z|x) || p(z) ), with p(z) = N(0, I).
        """
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

    # ====== GMM-BASED DENSITY & RARITY ======

    def fit_gmm(self, embeddings):
        """
        Fit a Gaussian Mixture Model (GMM) to the provided embeddings (mu).

        Args:
            embeddings (torch.Tensor or np.ndarray): shape (num_samples, embedding_dim)
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.gmm = GaussianMixture(
            n_components=self.n_components, covariance_type="full", random_state=42
        )
        self.gmm.fit(embeddings)

    def set_rare_threshold(self, embeddings, fraction=0.1):
        """
        Compute a log-probability threshold so that approximately `fraction` of embeddings
        lie below it -> define them as 'rare'.

        Args:
            embeddings (torch.Tensor or np.ndarray): shape (num_samples, embedding_dim)
            fraction (float): fraction of samples considered 'rare'. e.g. 0.1 => 10%
        """
        if self.gmm is None:
            raise ValueError("GMM is not fitted. Call `fit_gmm` first.")

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        log_probs = self.gmm.score_samples(embeddings)
        cutoff = np.percentile(log_probs, fraction * 100)
        self.log_prob_threshold = cutoff

    def compute_log_likelihood(self, embeddings):
        """
        Compute the log probability of each embedding under the GMM.

        Args:
            embeddings (torch.Tensor): shape (batch_size, embedding_dim)

        Returns:
            log_probs (torch.Tensor): shape (batch_size,)
        """
        if self.gmm is None:
            raise ValueError("GMM is not fitted. Call `fit_gmm` first.")

        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings

        # shape: (batch_size,)
        log_probs_np = self.gmm.score_samples(embeddings_np)
        return torch.from_numpy(log_probs_np).to(self.device)

    def is_rare(self, embeddings):
        """
        Check if each embedding is 'rare' based on whether its log-likelihood
        is below the configured threshold.

        Args:
            embeddings (torch.Tensor): shape (batch_size, embedding_dim)

        Returns:
            rare_mask (torch.BoolTensor): shape (batch_size,)
        """
        if self.log_prob_threshold is None:
            raise ValueError(
                "log_prob_threshold is not set. Call `set_rare_threshold` first."
            )

        log_probs = self.compute_log_likelihood(embeddings)
        rare_mask = log_probs < self.log_prob_threshold
        return rare_mask
