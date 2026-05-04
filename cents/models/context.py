import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Optional
from .context_registry import register_context_module

class BaseContextModule(nn.Module):
    """
    Base class for context modules. Subclasses must implement the forward method.
    """
    @abstractmethod
    def forward(self, context_vars: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass

@register_context_module("default", "mlp")
class MLPContextModule(BaseContextModule):
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
                name: nn.Embedding(num_categories[1], embedding_dim)
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
                var_name: nn.Linear(embedding_dim, num_categories[1])
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
        embeddings = []
        for name, layer in self.context_embeddings.items():
            idx = context_vars[name]
            if idx.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                idx = idx.clamp(0, layer.num_embeddings - 1)
            embeddings.append(layer(idx))

        context_matrix = torch.cat(embeddings, dim=1)
        embedding = self.mlp(context_matrix)

        classification_logits = {
            var_name: head(embedding)
            for var_name, head in self.classification_heads.items()
        }

        return embedding, classification_logits

@register_context_module("default", "sep_mlp")
class SepMLPContextModule(BaseContextModule):
    def __init__(
        self, 
        context_vars: dict[str, int], 
        embedding_dim: int, 
        init_depth: int = 1, 
        mixing_depth: int = 1, 
    ) -> None:
        """
        Initialize SepMLPContextModule.
        
        Args:
            context_vars: Mapping of variable names to category counts.
            embedding_dim: Size of embedding vectors.
            init_depth: Depth of initial MLPs.
            mixing_depth: Depth of mixing MLP.
            continuous_vars: List of continuous variable names.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.continuous_vars = [k for k, v in context_vars.items() if v[0] == "continuous"]
        self.categorical_vars = {k: v[1] for k, v in context_vars.items() if v[0] == "categorical"}
        self.context_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_categories, embedding_dim)
                for name, num_categories in self.categorical_vars.items()
            }
        )

        # For continuous variables, use a simple linear projection
        self.continuous_projections = nn.ModuleDict(
            {
                name: nn.Linear(1, embedding_dim)
                for name in self.continuous_vars
            }
        )

        self.init_mlps = nn.ModuleDict({
            name: nn.Sequential(*[
                layer
                for _ in range(init_depth)
                for layer in (nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
            ])
            for name in self.categorical_vars.keys()
        })

        # Also create init MLPs for continuous variables
        self.continuous_init_mlps = nn.ModuleDict({
            name: nn.Sequential(*[
                layer
                for _ in range(init_depth)
                for layer in (nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
            ])
            for name in self.continuous_vars
        })

        total_dim = embedding_dim * (len(self.categorical_vars) + len(self.continuous_vars))

        self.mixing_mlp = nn.Sequential(            
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim))

        self.classification_heads = nn.ModuleDict(
            {
                var_name: nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, num_categories)
                )
                for var_name, num_categories in self.categorical_vars.items()
            }
        )
        
        # Regression heads for continuous variables (output single value for MSE loss)
        self.regression_heads = nn.ModuleDict(
            {
                var_name: nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, 1)
                )
                for var_name in self.continuous_vars
            }
        )

    def forward(self, context_vars):
        encodings = {}
        
        # Process categorical variables (only those present in context_vars)
        for name, layer in self.context_embeddings.items():
            if name in context_vars:
                idx = context_vars[name]
                if idx.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                    idx = idx.clamp(0, layer.num_embeddings - 1)
                encodings[name] = layer(idx)
        
        # Process continuous variables (only those present in context_vars)
        for name, layer in self.continuous_projections.items():
            if name in context_vars:
                continuous_val = context_vars[name].float()
                # DataLoader stacks 0-dim scalars into (batch,); layer expects (batch, 1)
                if continuous_val.dim() == 1:
                    continuous_val = continuous_val.unsqueeze(-1)
                encodings[name] = layer(continuous_val)

        embeddings = []        
        # Apply init MLPs to categorical variables
        for name, layer in self.init_mlps.items():
            if name in encodings:
                embeddings.append(layer(encodings[name]))
        
        # Apply init MLPs to continuous variables
        for name, layer in self.continuous_init_mlps.items():
            if name in encodings:
                embedding_output = layer(encodings[name])
                # Check for NaN in embedding output
                if torch.isnan(embedding_output).any():
                    raise ValueError(
                        f"NaN detected in embedding output for continuous variable '{name}' "
                        f"after init MLP. This may indicate numerical instability in the MLP layers."
                    )
                embeddings.append(embedding_output)

        context_matrix = torch.cat(embeddings, dim=1)
        embedding = self.mixing_mlp(context_matrix)

        classification_logits = {
            var_name: head(embedding)
            for var_name, head in self.classification_heads.items()
            if var_name in context_vars
        }
        
        # Regression outputs for continuous variables
        regression_outputs = {
            var_name: head(embedding).squeeze(-1)  # Remove last dim to get (batch_size,)
            for var_name, head in self.regression_heads.items()
            if var_name in context_vars
        }
        
        all_outputs = {**classification_logits, **regression_outputs}

        return embedding, all_outputs


@register_context_module("transformer")
class TransformerStaticContextModule(BaseContextModule):
    """
    Transformer-based static context embedder.

    Each context variable is projected to a token of size embedding_dim,
    augmented with a per-variable type embedding, then normalised before
    being fed into a shared Transformer encoder.  Mean-pooling across the
    variable tokens produces the final (B, embedding_dim) conditioning vector.

    Compared to MLPContextModule:
    - Attention captures interactions between context variables
    - pre-LN (norm_first=True) and GELU throughout → more stable gradients
    - No hardcoded bottleneck; width is controlled by dim_feedforward
    """

    def __init__(
        self,
        context_vars: dict[str, list],
        embedding_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.continuous_vars = [k for k, v in context_vars.items() if v[0] == "continuous"]
        self.categorical_vars = {k: v[1] for k, v in context_vars.items() if v[0] == "categorical"}
        self.var_names = list(self.categorical_vars.keys()) + self.continuous_vars

        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(n_cats, embedding_dim)
            for name, n_cats in self.categorical_vars.items()
        })
        self.cont_projections = nn.ModuleDict({
            name: nn.Linear(1, embedding_dim)
            for name in self.continuous_vars
        })
        # Per-variable learnable offset so attention can distinguish variable identity
        self.type_embeddings = nn.Embedding(len(self.var_names), embedding_dim)
        self.register_buffer("_var_indices", torch.arange(len(self.var_names)))

        # Normalise tokens before encoder to equalize scales across embed/projection types
        self.token_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(embedding_dim)

        self.classification_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, n_cats),
            )
            for name, n_cats in self.categorical_vars.items()
        })
        self.regression_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, 1),
            )
            for name in self.continuous_vars
        })

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.type_embeddings.weight, std=0.02)

    def forward(self, context_vars: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        all_type_embs = self.type_embeddings(self._var_indices)  # (N_vars, D)
        tokens = []
        for i, name in enumerate(self.var_names):
            if name in self.categorical_vars:
                idx = context_vars[name]
                if idx.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                    idx = idx.clamp(0, self.cat_embeddings[name].num_embeddings - 1)
                tok = self.cat_embeddings[name](idx) + all_type_embs[i]
            else:
                val = context_vars[name].float()
                if val.dim() == 1:
                    val = val.unsqueeze(-1)
                tok = self.cont_projections[name](val) + all_type_embs[i]
            tokens.append(self.token_norm(tok))

        x = torch.stack(tokens, dim=1)          # (B, N_vars, D)
        x = self.encoder(x)                     # (B, N_vars, D)
        embedding = self.output_norm(x.mean(dim=1))  # (B, D)

        classification_logits = {
            name: head(embedding)
            for name, head in self.classification_heads.items()
            if name in context_vars
        }
        regression_outputs = {
            name: head(embedding).squeeze(-1)
            for name, head in self.regression_heads.items()
            if name in context_vars
        }
        return embedding, {**classification_logits, **regression_outputs}


@register_context_module("dynamic_transformer")
class DynamicContextModule_Transformer(BaseContextModule):
    """
    Context module for processing dynamic (time series) context variables.
    Uses Transformer encoder to encode time series sequences into embeddings.

    Returns the full encoded sequence (B, T, embedding_dim) rather than a
    pooled vector, so temporal structure is preserved for cross-attention
    conditioning in the diffusion backbone.
    """

    # Signals to BaseModel that this module returns (B, T, emb_dim) not (B, emb_dim)
    returns_sequence = True

    def __init__(
        self,
        context_vars: dict[str, int],
        embedding_dim: int,
        seq_len: int = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
    ):
        """
        Initialize DynamicContextModule_Transformer.
        
        Args:
            context_vars: Mapping of variable names to category counts (for categorical time series)
                         or None (for numeric time series). Format: {name: [type, num_categories]}
            embedding_dim: Size of embedding vectors.
            seq_len: Sequence length of time series context variables.
            n_layers: Number of transformer encoder layers.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            dim_feedforward: Dimension of feedforward network in transformer.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        
        # Separate categorical and numeric time series
        self.categorical_ts_vars = {
            k: v[1] for k, v in context_vars.items() 
            if v[0] == "time_series" and v[1] is not None
        }
        self.numeric_ts_vars = [
            k for k, v in context_vars.items() 
            if v[0] == "time_series" and v[1] is None
        ]
        
        # For categorical time series, use embedding
        self.ts_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_categories, embedding_dim)
            for name, num_categories in self.categorical_ts_vars.items()
        })
        
        # For numeric time series, use linear projection to embedding_dim
        self.ts_projections = nn.ModuleDict({
            name: nn.Linear(1, embedding_dim)
            for name in self.numeric_ts_vars
        })
        
        # Positional encoding for transformer
        if seq_len is not None:
            self.pos_encodings = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1, seq_len, embedding_dim))
                for name in list(self.categorical_ts_vars.keys()) + self.numeric_ts_vars
            })
        else:
            self.pos_encodings = None
        
        # Transformer encoder for each time series variable
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        
        self.ts_encoders = nn.ModuleDict({
            name: nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            for name in list(self.categorical_ts_vars.keys()) + self.numeric_ts_vars
        })
        

        n_vars = len(self.categorical_ts_vars) + len(self.numeric_ts_vars)
        self.var_mix = nn.Linear(n_vars * embedding_dim, embedding_dim) if n_vars > 1 else None

        all_var_names = list(self.categorical_ts_vars.keys()) + self.numeric_ts_vars
        self.post_encoder_norms = nn.ModuleDict({
            name: nn.LayerNorm(embedding_dim)
            for name in all_var_names
        })
        self.post_mix_norm = nn.LayerNorm(embedding_dim)

        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using appropriate initialization strategies.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Parameter):
                if module.dim() == 3:  # (1, seq_len, embedding_dim)
                    nn.init.normal_(module, std=0.02)
    
    def forward(self, context_vars: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Process dynamic (time series) context variables using Transformer.

        Args:
            context_vars: Dict mapping variable names to tensors.
                         For categorical TS: (batch, seq_len) integer values.
                         For numeric TS: (batch, seq_len) float values.

        Returns:
            sequence: Combined sequence of shape (batch, seq_len, embedding_dim).
                      Temporal structure is preserved for downstream cross-attention.
            outputs: Empty dict for interface compatibility.
        """
        sequences = []

        # Process categorical time series
        for name in self.categorical_ts_vars.keys():
            if name in context_vars:
                ts_data = context_vars[name]  # (B, T)
                if torch.isnan(ts_data).any() or torch.isinf(ts_data).any():
                    raise ValueError(f"NaN/Inf in categorical TS input '{name}'")
                embedded = self.ts_embeddings[name](ts_data)  # (B, T, emb_dim)
                if self.pos_encodings is not None and name in self.pos_encodings:
                    embedded = embedded + self.pos_encodings[name][:, :embedded.size(1)]
                encoded = self.ts_encoders[name](embedded)  # (B, T, emb_dim)
                encoded = self.post_encoder_norms[name](encoded)
                if torch.isnan(encoded).any() or torch.isinf(encoded).any():
                    raise ValueError(f"NaN/Inf after transformer encoding '{name}'")
                sequences.append(encoded)

        # Process numeric time series
        for name in self.numeric_ts_vars:
            if name in context_vars:
                ts_data = context_vars[name]  # (B, T)
                if not ts_data.is_floating_point():
                    ts_data = ts_data.float()
                ts_data = torch.where(torch.isfinite(ts_data), ts_data, torch.zeros_like(ts_data))
                # Per-sample z-score normalisation over the time axis
                ts_mean = ts_data.mean(dim=1, keepdim=True)
                ts_std = ts_data.std(dim=1, keepdim=True) + 1e-8
                ts_data = (ts_data - ts_mean) / ts_std
                embedded = self.ts_projections[name](ts_data.unsqueeze(-1))  # (B, T, emb_dim)
                if self.pos_encodings is not None and name in self.pos_encodings:
                    embedded = embedded + self.pos_encodings[name][:, :embedded.size(1)]
                if torch.isnan(embedded).any() or torch.isinf(embedded).any():
                    raise ValueError(f"NaN/Inf after projection for '{name}'")
                encoded = self.ts_encoders[name](embedded)  # (B, T, emb_dim)
                encoded = self.post_encoder_norms[name](encoded)
                if torch.isnan(encoded).any() or torch.isinf(encoded).any():
                    raise ValueError(f"NaN/Inf after transformer encoding numeric TS '{name}'")
                sequences.append(encoded)

        if not sequences:
            device = next(iter(context_vars.values())).device if context_vars else None
            batch_size = next(iter(context_vars.values())).size(0) if context_vars else 1
            T = self.seq_len if self.seq_len is not None else 1
            return torch.zeros(batch_size, T, self.embedding_dim, device=device), {}

        if len(sequences) == 1:
            out = sequences[0]
        elif self.var_mix is not None:
            # Learned combination across variables: concat along feature axis, project back
            out = self.var_mix(torch.cat(sequences, dim=-1))  # (B, T, emb_dim)
        else:
            # Single-variable fallback (var_mix is None only when n_vars == 1)
            out = sequences[0]

        out = self.post_mix_norm(out)

        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("NaN/Inf in dynamic context sequence output")

        return out, {}


@register_context_module("dynamic_joint_transformer")
class DynamicContextModule_JointTransformer(BaseContextModule):
    """
    Joint multi-channel encoder for dynamic context.

    All numeric time-series variables are stacked as channels (B, T, n_vars),
    projected jointly to (B, T, embedding_dim), then encoded by a single shared
    Transformer. Self-attention operates across time steps while seeing all
    variables simultaneously, allowing it to learn non-linear variable
    interactions (e.g. high TI + low wind → elevated PM2.5).

    Compare to DynamicContextModule_Transformer which runs one independent
    transformer per variable and combines them with a linear mix — that
    architecture can only learn additive contributions.
    """

    returns_sequence = True

    def __init__(
        self,
        context_vars: dict,
        embedding_dim: int,
        seq_len: int = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        self.numeric_ts_vars = [
            k for k, v in context_vars.items()
            if v[0] == "time_series" and v[1] is None
        ]
        self.categorical_ts_vars = {
            k: v[1] for k, v in context_vars.items()
            if v[0] == "time_series" and v[1] is not None
        }

        n_numeric = len(self.numeric_ts_vars)

        if n_numeric > 0:
            self.numeric_input_proj = nn.Linear(n_numeric, embedding_dim)
        else:
            self.numeric_input_proj = None

        # Categorical time-series: embed each to emb_dim and add into the joint repr
        self.ts_cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_categories, embedding_dim)
            for name, num_categories in self.categorical_ts_vars.items()
        })

        if seq_len is not None:
            self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))
        else:
            self.pos_encoding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.post_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        if self.pos_encoding is not None:
            nn.init.normal_(self.pos_encoding, std=0.02)

    def forward(self, context_vars: dict) -> tuple:
        numeric_tensors = []
        for name in self.numeric_ts_vars:
            if name in context_vars:
                ts = context_vars[name]
                if not ts.is_floating_point():
                    ts = ts.float()
                ts = torch.where(torch.isfinite(ts), ts, torch.zeros_like(ts))
                ts_mean = ts.mean(dim=1, keepdim=True)
                ts_std = ts.std(dim=1, keepdim=True) + 1e-8
                ts = (ts - ts_mean) / ts_std
                numeric_tensors.append(ts)

        if numeric_tensors and self.numeric_input_proj is not None:
            # (B, T, n_vars) → (B, T, emb_dim)
            x = self.numeric_input_proj(torch.stack(numeric_tensors, dim=-1))
        else:
            device = next(iter(context_vars.values())).device
            B = next(iter(context_vars.values())).size(0)
            T = self.seq_len or 1
            x = torch.zeros(B, T, self.embedding_dim, device=device)

        for name, emb_layer in self.ts_cat_embeddings.items():
            if name in context_vars:
                x = x + emb_layer(context_vars[name])  # (B, T, emb_dim)

        if self.pos_encoding is not None:
            x = x + self.pos_encoding[:, :x.size(1)]

        x = self.encoder(x)
        x = self.post_norm(x)

        return x, {}