import torch

from generator.config import load_model_config


class Options:
    """
    Class to handle model-specific configuration options.

    Args:
        model_name (str): The name of the model to load parameters for. This can be one of the predefined models
                          such as "diffcharge", "diffusion_ts", or "acgan".

    Attributes:
        seed (int): Random seed for reproducibility.
        model_name (str): The name of the model.
        device (torch.device): Device on which to run the model (CUDA or CPU).
        batch_size (int): Batch size for training.
        seq_len (int): Sequence length for the time series.
        input_dim (int): Input dimension of the data.
        noise_dim (int): Dimension of the noise input.
        cond_emb_dim (int): Dimension of the conditional embedding (if applicable).
        shuffle (bool): Whether to shuffle the data.
        (Additional attributes specific to different models are dynamically added.)
    """

    def __init__(self, model_name: str):
        config = load_model_config()
        self.seed = 42
        self.model_name = model_name
        self.device = config.device

        self.seq_len = config.seq_len
        self.input_dim = config.input_dim
        self.noise_dim = config.noise_dim
        self.cond_emb_dim = config.cond_emb_dim
        self.shuffle = config.shuffle
        self.sparse_conditioning_loss_weight = config.sparse_conditioning_loss_weight
        self.freeze_cond_after_warmup = config.freeze_cond_after_warmup
        self.categorical_dims = config.get("conditioning_vars", {})

        if model_name == "diffcharge":
            self._load_diffcharge_params(config["diffcharge"])
        elif model_name == "diffusion_ts":
            self._load_diffusion_ts_params(config["diffusion_ts"])
        elif model_name == "acgan":
            self._load_acgan_params(config["acgan"])

    def _load_diffcharge_params(self, model_params):
        """
        Load parameters specific to the "diffcharge" model.

        Args:
            model_params: Configuration dictionary for the diffcharge model.
        """
        self.batch_size = model_params.batch_size
        self.n_epochs = model_params.n_epochs
        self.init_lr = model_params.init_lr
        self.network = model_params.network
        self.guidance_scale = model_params.guidance_scale
        self.hidden_dim = model_params.hidden_dim
        self.nhead = model_params.nhead
        self.beta_start = model_params.beta_start
        self.beta_end = model_params.beta_end
        self.n_steps = model_params.n_steps
        self.schedule = model_params.schedule
        self.warm_up_epochs = model_params.warm_up_epochs

    def _load_diffusion_ts_params(self, model_params):
        """
        Load parameters specific to the "diffusion_ts" model.

        Args:
            model_params: Configuration dictionary for the diffusion_ts model.
        """
        self.batch_size = model_params.batch_size
        self.n_epochs = model_params.n_epochs
        self.n_steps = model_params.n_steps
        self.base_lr = model_params.base_lr
        self.n_layer_enc = model_params.n_layer_enc
        self.n_layer_dec = model_params.n_layer_dec
        self.d_model = model_params.d_model
        self.sampling_timesteps = model_params.sampling_timesteps
        self.loss_type = model_params.loss_type
        self.beta_schedule = model_params.beta_schedule
        self.n_heads = model_params.n_heads
        self.mlp_hidden_times = model_params.mlp_hidden_times
        self.eta = model_params.eta
        self.attn_pd = model_params.attn_pd
        self.resid_pd = model_params.resid_pd
        self.kernel_size = model_params.kernel_size
        self.padding_size = model_params.padding_size
        self.use_ff = model_params.use_ff
        self.reg_weight = model_params.reg_weight
        self.results_folder = model_params.results_folder
        self.gradient_accumulate_every = model_params.gradient_accumulate_every
        self.save_cycle = model_params.save_cycle
        self.ema_decay = model_params.ema_decay
        self.ema_update_interval = model_params.ema_update_interval
        self.lr_scheduler_params = model_params.lr_scheduler_params
        self.warm_up_epochs = model_params.warm_up_epochs
        self.use_ema_sampling = model_params.use_ema_sampling

    def _load_acgan_params(self, model_params):
        """
        Load parameters specific to the "acgan" model.

        Args:
            model_params: Configuration dictionary for the acgan model.
        """
        self.batch_size = model_params.batch_size
        self.n_epochs = model_params.n_epochs
        self.lr_gen = model_params.lr_gen
        self.lr_discr = model_params.lr_discr
        self.warm_up_epochs = model_params.warm_up_epochs
        self.include_auxiliary_losses = model_params.include_auxiliary_losses
        self.save_cycle = model_params.save_cycle
