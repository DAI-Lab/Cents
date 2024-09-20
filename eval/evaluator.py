from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter

from eval.discriminative_metric import discriminative_score_metrics
from eval.metrics import Context_FID
from eval.metrics import calculate_mmd
from eval.metrics import calculate_period_bound_mse
from eval.metrics import dynamic_time_warping_dist
from eval.metrics import plot_range_with_syn_values
from eval.metrics import plot_syn_with_closest_real_ts
from eval.metrics import visualization
from eval.predictive_metric import predictive_score_metrics
from generator.diffcharge.diffusion import DDPM
from generator.diffusion_ts.gaussian_diffusion import Diffusion_TS
from generator.gan.acgan import ACGAN
from generator.llm.llm import GPT
from generator.llm.llm import HF
from generator.options import Options

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    """
    A class for evaluating generative models on time series data.

    This class provides methods to evaluate different generative models on real datasets,
    compute various metrics, and visualize the results.

    Attributes:
        real_dataset: The real dataset used for evaluation.
        model_name (str): The name of the generative model being evaluated.
        writer (SummaryWriter): TensorBoard writer for logging evaluation results.
        metrics (Dict[str, List]): Dictionary to store computed metrics for each evaluation.
    """

    def __init__(self, real_dataset: Any, model_name: str, log_dir: str = "runs"):
        """
        Initialize the Evaluator.

        Args:
            real_dataset: The real dataset used for evaluation.
            model_name (str): The name of the generative model being evaluated.
            log_dir (str): Directory for storing TensorBoard logs.
        """
        self.real_dataset = real_dataset
        self.model_name = model_name
        self.writer = SummaryWriter(f"{log_dir}/{model_name}")
        self.metrics: Dict[str, List] = {
            "dtw": [],
            "mmd": [],
            "mse": [],
            "fid": [],
            "discr": [],
            "pred": [],
        }

    def evaluate_for_user(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the model for a specific user.

        Args:
            user_id (int): The ID of the user to evaluate.
            differenced (bool): Whether to use first-differenced data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Synthetic and real data for the user.
        """
        user_dataset = self.real_dataset.create_user_dataset(user_id)
        model = self.get_trained_model_for_user(self.model_name, user_dataset)
        user_log_dir = f"{self.writer.log_dir}/user_{user_id}"
        user_writer = SummaryWriter(user_log_dir)

        print("----------------------")
        print(f"Starting evaluation for user {user_id}")
        print("----------------------")

        return self.run_eval(user_dataset, model, user_writer, user_id)

    def run_eval(
        self, dataset: Any, model: Any, writer: SummaryWriter, user_id: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the evaluation for a given dataset and model.

        Args:
            dataset: The dataset to evaluate.
            model: The trained model to evaluate.
            writer (SummaryWriter): TensorBoard writer for logging results.
            user_id (int, optional): The ID of the user being evaluated.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Synthetic and real data arrays.
        """
        real_user_data = dataset.data
        syn_user_data, real_user_data = self.generate_dataset_for_eval(
            model, dataset.data
        )
        syn_user_data.reset_index()["dataid"] = real_user_data.reset_index()["dataid"]

        real_user_data_inv = dataset.inverse_transform(real_user_data)
        syn_user_data_inv = dataset.inverse_transform(syn_user_data)

        real_data_array = np.stack(real_user_data["timeseries"])
        syn_data_array = np.stack(syn_user_data["timeseries"])
        real_data_array_inv = np.stack(real_user_data_inv["timeseries"])
        syn_data_array_inv = np.stack(syn_user_data_inv["timeseries"])

        # Compute metrics
        self._compute_dtw(real_data_array_inv, syn_data_array_inv, writer, user_id)
        self._compute_mmd(real_data_array_inv, syn_data_array_inv, writer, user_id)
        self._compute_mse(syn_data_array_inv, real_user_data_inv, writer, user_id)
        self._compute_fid(real_data_array, syn_data_array, writer, user_id)
        self._compute_discriminative_score(
            real_data_array_inv, syn_data_array_inv, writer, user_id
        )
        self._compute_predictive_score(
            real_data_array_inv, syn_data_array_inv, writer, user_id
        )

        # Visualization
        self._create_visualizations(
            real_user_data,
            real_user_data_inv,
            syn_user_data_inv,
            dataset,
            model,
            writer,
        )

        writer.flush()
        writer.close()
        return syn_data_array_inv[:, :, 0], real_data_array_inv[:, :, 0]

    def evaluate_all_user_models(self):
        """
        Evaluate the model for all users in the dataset.
        """
        user_ids = self.real_dataset.data["dataid"].unique()
        syn_data = []
        real_data = []

        for user_id in user_ids:
            if user_id == 3687:
                syn_user_data, real_user_data = self.evaluate_for_user(user_id)
                syn_data.append(syn_user_data)
                real_data.append(real_user_data)

        syn_data = np.expand_dims(np.concatenate(syn_data, axis=0), axis=-1)
        real_data = np.expand_dims(np.concatenate(real_data, axis=0), axis=-1)

        plot = visualization(real_data, syn_data, "tsne")
        self.writer.add_figure(tag=f"TSNE", figure=plot)

        self._log_final_results()

    def generate_samples_for_eval(
        self, dataid: int, model: Any, dataset: Any, num_samples: int
    ) -> pd.DataFrame:
        """
        Generate synthetic samples for evaluation.

        Args:
            dataid (int): The ID of the data point.
            model: The trained model to generate samples.
            dataset: The original dataset to ensure that only contained conditional variables are generated.
            num_samples (int): The number of samples to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the generated samples.
        """
        random_conditioning_vars = model.sample_conditioning_vars(
            dataset, 1, random=False
        )

        for keys, tensor in random_conditioning_vars.items():
            random_conditioning_vars[keys] = tensor.repeat(
                num_samples
            )  # repeat tensors according to specified num_samples

        generated_ts = model.generate(random_conditioning_vars).cpu().numpy()

        if len(generated_ts.shape) == 2:
            generated_ts = generated_ts.reshape(
                generated_ts.shape[0], -1, generated_ts.shape[1]
            )

        syn_ts_df = pd.DataFrame()

        for key, tensor in random_conditioning_vars.items():
            syn_ts_df[key] = tensor.cpu().numpy().flatten()
        syn_ts_df["timeseries"] = list(generated_ts)
        syn_ts_df["dataid"] = dataid
        return syn_ts_df

    def generate_dataset_for_eval(
        self,
        model: Any,
        real_user_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a synthetic dataset for evaluation, using a stratified subset.

        Args:
            model: The trained model to generate samples.
            real_user_df (pd.DataFrame): The real user data.

        Returns:
            pd.DataFrame: A DataFrame containing the generated dataset.
            pd.DataFrame: Chosen subset of the real dataset
        """
        y = real_user_df["dataid"]

        if y.nunique() > 1:  # if there are multiple users, use a stratified subset
            num_samples = real_user_df.shape[0] // 10
            strat_split = StratifiedShuffleSplit(
                n_splits=1, test_size=num_samples, random_state=42
            )

            for _, subset_index in strat_split.split(real_user_df, y):
                subset_real_data = real_user_df.iloc[subset_index]

            real_user_df = subset_real_data.reset_index(drop=True)

        real_conditioning_vars = {
            name: torch.tensor(real_user_df[name].values, dtype=torch.long)
            for name in model.opt.categorical_dims.keys()
        }

        generated_ts = model.generate(real_conditioning_vars).cpu().numpy()

        if len(generated_ts.shape) == 2:
            generated_ts = generated_ts.reshape(
                generated_ts.shape[0], -1, generated_ts.shape[1]
            )

        syn_ts = real_user_df.copy()
        syn_ts["timeseries"] = list(generated_ts)

        return syn_ts, real_user_df

    def evaluate_all_pv_users(self):
        """
        Evaluate the model for all PV users in the dataset.
        """
        dataset = self.real_dataset.create_all_pv_user_dataset()
        model = self.get_trained_model_for_user(self.model_name, dataset)
        user_log_dir = f"{self.writer.log_dir}/pv_users"
        user_writer = SummaryWriter(user_log_dir)

        print("----------------------")
        print("Starting evaluation for all PV users")
        print("----------------------")

        self.run_eval(dataset, model, user_writer, None)

    def evaluate_all_users(self):
        """
        Evaluate the model for all users in the same dataset.
        """
        dataset = self.real_dataset.create_all_user_dataset()
        dataset.is_pv_user = False
        model = self.get_trained_model_for_user(self.model_name, dataset)
        user_log_dir = f"{self.writer.log_dir}/all_users"
        user_writer = SummaryWriter(user_log_dir)

        print("----------------------")
        print("Starting evaluation for all users")
        print("----------------------")

        self.run_eval(dataset, model, user_writer, None)

    def evaluate_all_non_pv_users(self):
        """
        Evaluate the model for all non-PV users in the dataset.
        """
        dataset = self.real_dataset.create_non_pv_user_dataset()
        model = self.get_trained_model_for_user(self.model_name, dataset)
        user_log_dir = f"{self.writer.log_dir}/non_pv_users"
        user_writer = SummaryWriter(user_log_dir)

        print("----------------------")
        print("Starting evaluation for all non-PV users")
        print("----------------------")

        self.run_eval(dataset, model, user_writer, None)

    def get_summary_metrics(self) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate the mean values for all users across all metrics.

        Returns:
            Tuple[float, float, float, float, float, float]: Mean values for DTW, MMD, MSE, FID,
            discriminative score, and predictive score.
        """
        metrics_summary = {
            metric: np.mean([value[1] for value in self.metrics[metric]])
            for metric in self.metrics.keys()
        }

        return (
            metrics_summary["dtw"],
            metrics_summary["mmd"],
            metrics_summary["mse"],
            metrics_summary["fid"],
            metrics_summary["discr"],
            metrics_summary["pred"],
        )

    def get_trained_model_for_user(self, model_name: str, user_dataset: Any) -> Any:
        """
        Get a trained model for a specific user dataset.

        Args:
            model_name (str): The name of the model to train.
            user_dataset: The user dataset to train the model on.

        Returns:
            Any: The trained model.

        Raises:
            ValueError: If the model name is not recognized.
        """
        input_dim = int(user_dataset.is_pv_user) + 1

        opt = Options(model_name)
        opt.input_dim = input_dim

        model_dict = {
            "acgan": ACGAN,
            "diffcharge": DDPM,
            "diffusion_ts": Diffusion_TS,
            "mistral": lambda opt: HF("mistralai/Mistral-7B-Instruct-v0.2"),
            "llama": lambda opt: HF("meta-llama/Meta-Llama-3.1-8B"),
            "gpt": lambda opt: GPT("gpt-4o"),
        }

        if model_name in model_dict:
            model_class = model_dict[model_name]
            model = model_class(opt) if not callable(model_class) else model_class(opt)
        else:
            raise ValueError("Model name not recognized!")

        model.train_model(user_dataset)
        return model

    def _compute_dtw(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        writer: SummaryWriter,
        user_id: int,
    ):
        """Compute Dynamic Time Warping distance."""
        dtw_mean, dtw_std = dynamic_time_warping_dist(real_data, syn_data)
        writer.add_scalar("DTW/mean", dtw_mean)
        writer.add_scalar("DTW/std", dtw_std)
        self.metrics["dtw"].append((user_id, dtw_mean, dtw_std))

    def _compute_mmd(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        writer: SummaryWriter,
        user_id: int,
    ):
        """Compute Maximum Mean Discrepancy."""
        mmd_mean, mmd_std = calculate_mmd(real_data, syn_data)
        writer.add_scalar("MMD/mean", mmd_mean)
        writer.add_scalar("MMD/std", mmd_std)
        self.metrics["mmd"].append((user_id, mmd_mean, mmd_std))

    def _compute_mse(
        self,
        syn_data: np.ndarray,
        real_data: pd.DataFrame,
        writer: SummaryWriter,
        user_id: int,
    ):
        """Compute Period Bound Mean Squared Error."""
        mse_mean, mse_std = calculate_period_bound_mse(syn_data, real_data)
        writer.add_scalar("MSE/mean", mse_mean)
        writer.add_scalar("MSE/std", mse_std)
        self.metrics["mse"].append((user_id, mse_mean, mse_std))

    def _compute_fid(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        writer: SummaryWriter,
        user_id: int,
    ):
        """Compute Context Fréchet Inception Distance."""
        print(f"Training TS2Vec for user {user_id}...")
        fid_score = Context_FID(real_data, syn_data)
        print("Done!")
        writer.add_scalar("FID/score", fid_score)
        self.metrics["fid"].append((user_id, fid_score))

    def _compute_discriminative_score(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        writer: SummaryWriter,
        user_id: int,
    ):
        """Compute discriminative score."""
        print(f"Computing discriminative score for user {user_id}...")
        discr_score, _, _ = discriminative_score_metrics(real_data, syn_data)
        print("Done!")
        writer.add_scalar("Discr/score", discr_score)
        self.metrics["discr"].append((user_id, discr_score))

    def _compute_predictive_score(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        writer: SummaryWriter,
        user_id: int,
    ):
        """Compute predictive score."""
        print(f"Computing predictive score for user {user_id}...")
        pred_score = predictive_score_metrics(real_data, syn_data)
        print("Done!")
        writer.add_scalar("Pred/score", pred_score)
        self.metrics["pred"].append((user_id, pred_score))

    def _create_visualizations(
        self,
        real_user_data: pd.DataFrame,
        real_user_data_inv: pd.DataFrame,
        syn_user_data_inv: pd.DataFrame,
        dataset: Any,
        model: Any,
        writer: SummaryWriter,
    ):
        """
        Create various visualizations for the evaluation results.

        Args:
            real_user_data (pd.DataFrame): Original real user data.
            real_user_data_inv (pd.DataFrame): Inverse-transformed real user data.
            syn_user_data_inv (pd.DataFrame): Inverse-transformed synthetic user data.
            dataset (Any): The dataset object.
            model (Any): The trained model.
            writer (SummaryWriter): TensorBoard writer for logging visualizations.
        """
        samples = self.generate_samples_for_eval(
            real_user_data["dataid"].iloc[0],
            model,
            dataset,
            num_samples=100,
        )
        samples = dataset.inverse_transform(samples)
        month = samples.iloc[0]["month"]
        weekday = samples.iloc[0]["weekday"]

        # Visualization 1: Plot range with synthetic values
        range_plot = plot_range_with_syn_values(
            real_user_data_inv, samples, month, weekday
        )
        writer.add_figure("Visualizations/Range_Plot", range_plot)

        # Visualization 2: Plot closest real signals with synthetic values
        closest_plot = plot_syn_with_closest_real_ts(
            real_user_data_inv, samples, month, weekday
        )
        writer.add_figure("Visualizations/Closest_Real_TS", closest_plot)

        # Visualization 4: t-SNE visualization of real and synthetic data
        real_data_array = np.stack(real_user_data_inv["timeseries"])
        syn_data_array = np.stack(syn_user_data_inv["timeseries"])
        # tsne_plot = visualization(real_data_array, syn_data_array, "tsne")
        # writer.add_figure("Visualizations/TSNE", tsne_plot)

        # Visualization 5: KDE plots for real and synthetic data
        kde_plot = visualization(real_data_array, syn_data_array, "kernel")
        writer.add_figure("Visualizations/KDE", kde_plot)

    def _log_final_results(self):
        """
        Log the final evaluation results.
        """
        (
            dtw_mean,
            mmd_mean,
            mse_mean,
            fid_mean,
            discr_mean,
            pred_mean,
        ) = self.get_summary_metrics()

        self.writer.add_scalar("Final_Results/DTW", dtw_mean)
        self.writer.add_scalar("Final_Results/MMD", mmd_mean)
        self.writer.add_scalar("Final_Results/MSE", mse_mean)
        self.writer.add_scalar("Final_Results/FID", fid_mean)
        self.writer.add_scalar("Final_Results/Discriminative_Score", discr_mean)
        self.writer.add_scalar("Final_Results/Predictive_Score", pred_mean)

        print("Final Evaluation Results:")
        print(f"DTW: {dtw_mean:.4f}")
        print(f"MMD: {mmd_mean:.4f}")
        print(f"MSE: {mse_mean:.4f}")
        print(f"FID: {fid_mean:.4f}")
        print(f"Discriminative Score: {discr_mean:.4f}")
        print(f"Predictive Score: {pred_mean:.4f}")

        self.writer.close()
