import torch


class Options:
    def __init__(self, model_name):
        self.seed = 42
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8  #
        self.seq_len = 96  # 96 for pecanstreet
        self.input_dim = 2  # 1
        self.noise_dim = 256
        self.cond_emb_dim = 32
        self.shuffle = True
        if model_name == "diffcharge":
            self.n_epochs = 1000
            self.init_lr = 1e-4
            self.network = "cnn"  # "attention" or "cnn"
            self.guidance_scale = 1.0
            self.hidden_dim = 256
            self.cond_emb_dim = 32
            self.nhead = 8
            self.beta_start = 1e-4
            self.beta_end = 0.02
            self.n_steps = 1000
            self.schedule = "linear"  # "cosine" # "linear"  # "quadratic"
        elif model_name == "diffusion_ts":
            self.batch_size = 32
            self.n_epochs = 1000
            self.n_steps = 1000
            self.base_lr = 1e-4
            self.n_layer_enc = 3
            self.n_layer_dec = 4
            self.d_model = 128
            self.cond_emb_dim = self.d_model
            self.sampling_timesteps = None
            self.loss_type = "l2"
            self.beta_schedule = "cosine"
            self.n_heads = 4
            self.mlp_hidden_times = 4
            self.eta = 0.0
            self.attn_pd = 0.0
            self.resid_pd = 0.0
            self.kernel_size = None
            self.padding_size = None
            self.use_ff = True
            self.reg_weight = None
            self.results_folder = "./Checkpoints_syn"
            self.gradient_accumulate_every = 2
            self.save_cycle = 1000
            self.ema_decay = 0.99
            self.ema_update_interval = 10
            self.lr_scheduler_params = {
                "factor": 0.5,
                "patience": 200,
                "min_lr": 1.0e-5,
                "threshold": 1.0e-1,
                "threshold_mode": "rel",
                "verbose": False,
            }

        elif model_name == "acgan":
            self.n_epochs = 200
            self.validate = False
            self.lr_gen = 1e-4
            self.lr_discr = 1e-4
