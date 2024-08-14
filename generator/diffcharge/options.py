import torch


class Options:
    def __init__(self, model_name, isTrain):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = 1000
        self.level = "station"  # ***
        self.seq_len = 96  # station——288, driver——720
        self.cond_flag = "conditional"  # ***
        if isTrain:
            self.batch_size = 8  # station——4, driver——8
            self.shuffle = True
        else:
            self.batch_size = 1
            self.shuffle = False
        if model_name == "diffusion":
            self.init_lr = 5e-5
            self.network = "cnn"  # "attention" or "cnn"
            self.input_dim = 2
            self.hidden_dim = 256
            self.cond_dim = 256
            self.nhead = 8
            self.beta_start = 1e-4
            self.beta_end = 0.02
            self.n_steps = 1000
            self.schedule = "linear"  # "cosine" # "linear"  # "quadratic"
