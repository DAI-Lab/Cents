import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data_utils.dataset import prepare_dataloader
from eval.loss import mmd_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.init as init


class RNNModule(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        month_size,
        weekday_size,
        output_size=None,
        rnn_type="lstm",
        apply_sigmoid=False,
    ):
        super(RNNModule, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.month_embedding = nn.Embedding(12, month_size).to(device)
        self.weekday_embedding = nn.Embedding(7, weekday_size).to(device)

        condition_size = month_size + weekday_size
        self.condition_linear = nn.Linear(condition_size, hidden_size)

        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size + hidden_size, hidden_size, num_layers, batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                input_size + hidden_size, hidden_size, num_layers, batch_first=True
            )
        self.fc = nn.Linear(hidden_size, output_size if output_size else hidden_size)

    def forward(self, x, month, weekday):
        batch_size, seq_len, _ = x.size()
        month_emb = self.month_embedding(month)
        weekday_emb = self.weekday_embedding(weekday)
        condition = torch.cat([month_emb, weekday_emb], dim=-1)

        # Pass through linear layer
        condition_hidden = self.condition_linear(condition)

        # Repeat condition for each time step
        condition_repeated = condition_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate input and condition
        x_combined = torch.cat([x, condition_repeated], dim=-1)

        # Initial hidden state
        h_0 = condition_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = h_0.clone() if isinstance(self.rnn, nn.LSTM) else None

        if isinstance(self.rnn, nn.LSTM):
            packed_output, (h_n, c_n) = self.rnn(x_combined, (h_0, c_0))
        else:
            packed_output, h_n = self.rnn(x_combined, h_0)

        output = self.fc(packed_output)

        if self.apply_sigmoid:
            output = torch.sigmoid(output)

        return output, (h_n, c_n) if isinstance(self.rnn, nn.LSTM) else (h_n,)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)


class TimeGAN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        month_size=32,
        weekday_size=32,
        rnn_type="gru",
    ):
        super(TimeGAN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.month_size = month_size
        self.weekday_size = weekday_size

        self.embedder = RNNModule(
            input_size,
            hidden_size,
            3,
            month_size,
            weekday_size,
            hidden_size,
            "gru",
            apply_sigmoid=True,
        ).to(device)

        self.recovery = RNNModule(
            hidden_size,
            hidden_size,
            num_layers,
            month_size,
            weekday_size,
            input_size,
            rnn_type,
            apply_sigmoid=True,
        ).to(device)

        self.generator = RNNModule(
            input_size,
            hidden_size,
            num_layers,
            month_size,
            weekday_size,
            hidden_size,
            rnn_type,
            apply_sigmoid=True,
        ).to(device)

        self.supervisor = RNNModule(
            hidden_size,
            hidden_size,
            num_layers,
            month_size,
            weekday_size,
            hidden_size,
            rnn_type,
            apply_sigmoid=True,
        ).to(device)

        self.discriminator = RNNModule(
            hidden_size,
            hidden_size,
            num_layers,
            month_size,
            weekday_size,
            1,
            rnn_type,
            apply_sigmoid=True,
        ).to(device)

        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, RNNModule):
            m._weights_init(m)

    def forward(self, X, Z, month, weekday):
        H, _ = self.embedder(X, month, weekday)
        X_tilde, _ = self.recovery(H, month, weekday)

        E_hat, _ = self.generator(Z, month, weekday)
        H_hat, _ = self.supervisor(E_hat, month, weekday)
        H_hat_supervise, _ = self.supervisor(H, month, weekday)

        X_hat, _ = self.recovery(H_hat, month, weekday)

        Y_fake, _ = self.discriminator(H_hat, month, weekday)
        Y_real, _ = self.discriminator(H, month, weekday)
        Y_fake_e, _ = self.discriminator(E_hat, month, weekday)

        return (
            H,
            X_tilde,
            E_hat,
            H_hat,
            H_hat_supervise,
            X_hat,
            Y_fake,
            Y_real,
            Y_fake_e,
        )

    def train_model(
        self,
        x_train,
        x_val,
        batch_size_embedding=16,
        batch_size_supervised=32,
        batch_size_joint=4,
        num_epoch=5,
    ):
        summary_writer = SummaryWriter()
        optimizer_embedder = optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=5e-5,
            betas=(0.5, 0.999),
        )

        optimizer_generator = optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=2e-4,
            betas=(0.5, 0.999),
        )
        optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999)
        )

        criterion = nn.BCELoss()
        mse_loss = nn.MSELoss()

        train_loader_embedding = prepare_dataloader(x_train, batch_size_embedding)
        train_loader_supervised = prepare_dataloader(x_train, batch_size_supervised)
        train_loader_joint = prepare_dataloader(x_train, batch_size_joint)
        val_loader = prepare_dataloader(x_val, batch_size_joint)

        # Phase 1: Embedding network training
        self.embedding_network_training(
            train_loader_embedding,
            optimizer_embedder,
            mse_loss,
            num_epoch,
            summary_writer,
        )

        # Phase 2: Training with supervised loss only
        self.supervised_training(
            train_loader_supervised,
            optimizer_generator,
            mse_loss,
            num_epoch,
            summary_writer,
        )

        # Phase 3: Joint training
        self.joint_training(
            train_loader_joint,
            val_loader,
            optimizer_embedder,
            optimizer_generator,
            optimizer_discriminator,
            criterion,
            mse_loss,
            num_epoch,
            summary_writer,
        )

        print("Training complete")

    def embedding_network_training(
        self, train_loader, optimizer_embedder, mse_loss, num_epoch, summary_writer
    ):
        print("Start Embedding Network Training")

        step = 0

        for epoch in range(num_epoch):
            for i, (time_series_batch, month_label_batch, day_label_batch) in enumerate(
                tqdm(train_loader, desc=f"Embedding Epoch {epoch + 1}")
            ):
                X_mb = time_series_batch.to(device)
                month_label_batch = month_label_batch.to(device)
                day_label_batch = day_label_batch.to(device)

                # Forward pass
                H, X_tilde, _, _, _, _, _, _, _ = self.forward(
                    X_mb,
                    torch.zeros_like(X_mb),
                    month_label_batch,
                    day_label_batch,
                )

                # Embedder and Recovery loss
                E_loss_T0 = mse_loss(X_mb, X_tilde)
                E_loss0 = 10 * torch.sqrt(E_loss_T0)

                # Optimize Embedder and Recovery
                optimizer_embedder.zero_grad()
                E_loss0.backward()
                optimizer_embedder.step()

                summary_writer.add_scalars(
                    "data/embedding",
                    {"emb": E_loss0.item()},
                    global_step=step,
                )

                if i % 1000 == 0:
                    print(
                        f"Embedding Epoch [{epoch}/{num_epoch}], Step [{i}/{len(train_loader)}], E_loss: {E_loss0.item():.4f}"
                    )

                step += 1

        print("Finish Embedding Network Training")

    def supervised_training(
        self, train_loader, optimizer_generator, mse_loss, num_epoch, summary_writer
    ):
        print("Start Training with Supervised Loss Only")

        step = 0

        for epoch in range(num_epoch // 2):
            for i, (time_series_batch, month_label_batch, day_label_batch) in enumerate(
                tqdm(train_loader, desc=f"Supervised Epoch {epoch + 1}")
            ):
                X_mb = time_series_batch.to(device)
                month_label_batch = month_label_batch.to(device)
                day_label_batch = day_label_batch.to(device)

                # Forward pass
                H, _, _, H_hat, H_hat_supervise, _, _, _, _ = self.forward(
                    X_mb,
                    torch.zeros_like(X_mb),
                    month_label_batch,
                    day_label_batch,
                )

                # Supervised loss
                G_loss_S = mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :])

                # Optimize Generator and Supervisor
                optimizer_generator.zero_grad()
                G_loss_S.backward()
                optimizer_generator.step()

                summary_writer.add_scalars(
                    "data/supervised",
                    {"gen": G_loss_S.item()},
                    global_step=step,
                )

                if i % 1000 == 0:
                    print(
                        f"Supervised Epoch [{epoch}/{num_epoch}], Step [{i}/{len(train_loader)}], G_loss_S: {G_loss_S.item():.4f}"
                    )

                step += 1

        print("Finish Training with Supervised Loss Only")

    def joint_training(
        self,
        train_loader,
        val_loader,
        optimizer_embedder,
        optimizer_generator,
        optimizer_discriminator,
        criterion,
        mse_loss,
        num_epoch,
        summary_writer,
    ):
        print("Start Joint Training")
        gamma = 1
        step = 0

        for epoch in range(num_epoch):
            for i, (time_series_batch, month_label_batch, day_label_batch) in enumerate(
                tqdm(train_loader, desc=f"Joint Epoch {epoch + 1}")
            ):
                current_batch_size, seq_len, n_dim = time_series_batch.shape
                T_mb = torch.full((current_batch_size,), seq_len, dtype=torch.int32).to(
                    device
                )
                Z_mb = torch.rand(current_batch_size, seq_len, n_dim).to(device)

                X_mb = time_series_batch.to(device)
                month_label_batch = month_label_batch.to(device)
                day_label_batch = day_label_batch.to(device)

                # Train Generator and Embedder (multiple steps)
                for _ in range(2):

                    (
                        H,
                        X_tilde,
                        E_hat,
                        H_hat,
                        H_hat_supervise,
                        X_hat,
                        Y_fake,
                        Y_real,
                        Y_fake_e,
                    ) = self.forward(X_mb, Z_mb, month_label_batch, day_label_batch)

                    # Generator losses
                    G_loss_U = criterion(Y_fake, torch.ones_like(Y_fake))
                    G_loss_U_e = criterion(Y_fake_e, torch.ones_like(Y_fake_e))
                    G_loss_S = mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :])
                    G_loss_V1 = torch.mean(
                        torch.abs(torch.std(X_hat, dim=0) - torch.std(X_mb, dim=0))
                    )
                    G_loss_V2 = torch.mean(
                        torch.abs(torch.mean(X_hat, dim=0) - torch.mean(X_mb, dim=0))
                    )
                    G_loss_V = G_loss_V1 + G_loss_V2

                    G_loss = (
                        (G_loss_U + G_loss_U_e)
                        + 100 * torch.sqrt(G_loss_S)
                        + 100 * G_loss_V
                    )

                    # Embedder losses
                    E_loss_T0 = mse_loss(X_mb, X_tilde)
                    E_loss0 = 10 * torch.sqrt(E_loss_T0)
                    E_loss = E_loss0 + 0.1 * G_loss_S

                    optimizer_generator.zero_grad()
                    optimizer_embedder.zero_grad()

                    G_loss.backward(retain_graph=True)
                    # E_loss.backward()

                    optimizer_generator.step()
                    # optimizer_embedder.step()

                # Train Discriminator
                optimizer_discriminator.zero_grad()

                _, _, _, _, _, _, Y_fake, Y_real, Y_fake_e = self.forward(
                    X_mb, Z_mb, month_label_batch, day_label_batch
                )

                D_loss_real = criterion(Y_real, torch.ones_like(Y_real))
                D_loss_fake = criterion(Y_fake, torch.zeros_like(Y_fake))
                D_loss_fake_e = criterion(Y_fake_e, torch.zeros_like(Y_fake_e))

                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

                # Only update discriminator if loss is above threshold
                if D_loss.item() > 0.15:
                    D_loss.backward()
                    optimizer_discriminator.step()

                summary_writer.add_scalars(
                    "data/adversarial",
                    {
                        "gen": G_loss.item(),
                        "discr": D_loss.item(),
                    },
                    global_step=step,
                )

                summary_writer.add_scalars(
                    "data/generator",
                    {
                        "gen_u": G_loss_U.item(),
                        "gen_u_e": G_loss_U_e.item(),
                        "gen_v": G_loss_V.item(),
                        "gen_s": G_loss_S.item(),
                    },
                    global_step=step,
                )

                summary_writer.add_scalars(
                    "data/discriminator",
                    {
                        "real": D_loss_real.item(),
                        "fake": D_loss_fake.item(),
                        "fake_e": D_loss_fake_e.item(),
                    },
                    global_step=step,
                )

                if i % 1000 == 0:
                    print(
                        f"Joint Epoch [{epoch}/{num_epoch}], "
                        f"G_loss: {G_loss.item():.4f}, D_loss: {D_loss.item():.4f}, E_loss: {E_loss.item():.4f}"
                    )

                step += 1

            # Validate
            self.validate(val_loader, epoch)

    def validate(self, val_loader, num_epoch):
        self.eval()
        with torch.no_grad():
            total_mmd_loss = torch.zeros(self.input_size, device=device).cpu()
            num_batches = 0
            for time_series_batch, month_label_batch, day_label_batch in val_loader:
                time_series_batch = time_series_batch.to(device)
                month_label_batch = month_label_batch.to(device)
                day_label_batch = day_label_batch.to(device)
                Z_mb = torch.rand(
                    time_series_batch.size(0),
                    time_series_batch.size(1),
                    self.input_size,
                    device=device,
                )

                x_generated = self.forward(
                    time_series_batch, Z_mb, month_label_batch, day_label_batch
                )[
                    5
                ]  # X_hat

                mmd_values = np.zeros(
                    shape=(time_series_batch.shape[0], self.input_size)
                )

                for dim in range(self.input_size):
                    mmd_values[:, dim] = mmd_loss(
                        time_series_batch[:, dim, :].cpu().numpy(),
                        x_generated[:, dim, :].cpu().numpy(),
                    )

                batch_mmd_loss = np.mean(mmd_values, axis=0)
                total_mmd_loss += batch_mmd_loss
                num_batches += 1

            mean_mmd_loss = total_mmd_loss / num_batches
            print(f"Epoch [{num_epoch}], Mean MMD Loss: {mean_mmd_loss.cpu().numpy()}")
        self.train()

    def generate(self, month_labels, day_labels, seq_length=96):
        num_samples = day_labels.shape[0]
        noise = torch.randn((num_samples, seq_length, self.input_size)).to(device)
        return self._generate(
            noise, month_labels.clone().to(device), day_labels.clone().to(device)
        )

    def _generate(self, Z, month, weekday):
        self.generator.eval()
        with torch.no_grad():
            E_hat, _ = self.generator(Z, month, weekday)
            H_hat, _ = self.supervisor(E_hat, month, weekday)
            X_hat, _ = self.recovery(H_hat, month, weekday)
            return X_hat

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
