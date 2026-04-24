import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from cents.utils.config_loader import load_yaml, apply_overrides

from cents.datasets.timeseries_dataset import TimeSeriesDataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
# These are warnings for an error that is accounted for in the code
warnings.filterwarnings("ignore", category=RuntimeWarning) 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CommercialDataset(TimeSeriesDataset):
    def __init__(self, cfg: DictConfig = None,
                 overrides: Optional[List[str]] = None,
                 force_retrain_normalizer: bool = False,
                 run_dir: Optional[str] = None):
        
        """
        Initializes the commercial energy dataset.
        Original Dataset: https://github.com/buds-lab/building-data-genome-project-2
        Note: This uses the clean version of their data, which already has some preprocessing done.
        Args:
            cfg (Optional[DictConfig]): Override Hydra config; if None,
                load from `config/dataset/commercial.yaml`.
            overrides (Optional[List[str]]): Override Hydra config; if None,
                load from `config/dataset/commercial.yaml` and apply overrides.
        """

        if cfg is None:
            cfg = load_yaml(os.path.join(ROOT_DIR, "config", "dataset", "commercial.yaml"))
            if overrides:
                cfg = apply_overrides(cfg, overrides)

        self.cfg = cfg
        self.name = cfg.name
        self.normalize = cfg.normalize
        self.cfg.time_series_columns = ["energy_meter"]
        self.geography = cfg.geography

        self._load_data()
        
        self.time_series_column_names = self.cfg.time_series_columns
        self.time_series_dims = len(self.time_series_column_names)

        super().__init__(
            data=self.data,
            time_series_column_names=self.time_series_column_names,
            seq_len=self.cfg.seq_len,
            context_var_column_names=list(self.cfg.context_vars.keys()),
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            skip_heavy_processing=cfg.get('skip_heavy_processing', False),
            size=cfg.get('max_samples', None),
            force_retrain_normalizer=force_retrain_normalizer,
            run_dir=run_dir,
        )

    def _load_data(self):
        """
        Loads in metadata and data for the commercial energy dataset.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))

        metapath = os.path.join(base_path, "metadata.csv")
        if not os.path.exists(metapath):
            raise FileNotFoundError(f"Metadata file not found at {metapath}")
        metadata = pd.read_csv(metapath, usecols=self.cfg.metadata_columns)

        data_path = os.path.join(base_path, "electricity_cleaned.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        data = pd.read_csv(data_path)

        data = data.melt(
            id_vars="timestamp",          # keep timestamp as is
            var_name="dataid",            # old column names (id1, id2, etc.) become values in this column
            value_name="energy_meter"     # their values go here
        )

        data['site_id'] = data['dataid'].str.split('_').str[0]

        if self.geography:
            if self.geography not in metadata["site_id"].unique():
                raise ValueError(f"Geography {self.geography} not found in metadata")
            data = data[data["site_id"] == self.geography]
            metadata = metadata[metadata["site_id"] == self.geography]

        max_samples = self.cfg.get('max_samples', None)
        if max_samples is not None:
            unique_ids = data['dataid'].unique()
            # Each building produces ~700 daily sequences; 2x buffer accounts for filtering
            est_seqs_per_id = 700
            n_ids = min(len(unique_ids), max(1, int(np.ceil(max_samples * 2.0 / est_seqs_per_id))))
            kept_ids = np.random.choice(unique_ids, size=n_ids, replace=False)
            data = data[data['dataid'].isin(kept_ids)]
            print(f"CommercialDataset: subsampling {n_ids} building IDs (est. ~{n_ids * est_seqs_per_id} sequences) for max_samples={max_samples}")

        self.data = data
        self.metadata = metadata

    def _preprocess_data(self, data):
        """
        Creates sequences of seq_len and merges metadata. Removes any sequences with missing data.
        
        Args:
            data (pd.DataFrame): Raw DataFrame including 'energy_meter' values.
        
        Returns:
            pd.DataFrame: Metadata columns, datetime, year, month, weekday, date_day, and array-valued 'energy_meter'.
        """
        data = data.copy()

        data['datetime'] = pd.to_datetime(data['timestamp'])
        # data = data.dropna(subset=['energy_meter'])  # any NaN makes the day shorter -> filtered by size later
        data = data.sort_values(by=["dataid", "datetime"])

        # Extract date for grouping (groups all hourly measurements from same calendar day)
        data['date'] = data['datetime'].dt.date
        
        grouped = data.groupby(['dataid', 'date'])['energy_meter'].apply(np.asarray).reset_index()

        # grouped = grouped[grouped["energy_meter"].apply(len) == self.cfg.seq_len].reset_index(drop=True)
        grouped = grouped[grouped["energy_meter"].apply(lambda x: not np.isnan(x).any())]
        grouped['date'] = pd.to_datetime(grouped['date'])


        grouped['year'] = grouped['date'].dt.year
        grouped["month"] = grouped["date"].dt.month_name()
        grouped["weekday"] = grouped["date"].dt.day_name()
        grouped["date_day"] = grouped["date"].dt.day

        if grouped["energy_meter"].apply(lambda x: np.isnan(x).any()).any():
            raise ValueError("NaN values remain in grouped energy_meter sequences after filtering.")
        merged = pd.merge(grouped, self.metadata, how="left", left_on="dataid", right_on="building_id").drop(columns=["building_id"])
        merged.sort_values(by=["dataid", "date"], inplace=True)

        merged = self._handle_missing_data(merged)
        
        # Check if any NaN remains
        context_cols = [col for col in self.cfg.context_vars.keys() if col in merged.columns]
        if merged[context_cols].isna().sum().sum() > 0:
            print(f"Warning: {merged[context_cols].isna().sum().sum()} NaN values remain after handling")

        mask = merged["energy_meter"].apply(lambda x: x.std() < 0.01)

        merged = merged[~mask]

        return merged
    
    def _handle_missing_data(self, merged):
        """
        Fill NaNs using hierarchical and context-aware imputation.
        
        Strategy:
        1. Numeric: Group-based median (by site/building type), fallback to global median
        2. Categorical: Use hierarchical structure (e.g., sub_primaryspaceusage from primaryspaceusage)
        3. Last resort: Mode or 'unknown' category
        
        Args:
            merged (pd.DataFrame): Merged sequence+metadata rows.
        
        Returns:
            pd.DataFrame: Fully imputed DataFrame.
        """
        df = merged.copy()
        
        # Handle numeric columns with group-based imputation
        numeric_cols = self.cfg.get('numeric_cols', [])
        for col in numeric_cols:
            if col in df.columns and df[col].isna().any():
                # Try imputing based on similar buildings (same site_id and primaryspaceusage)
                if 'site_id' in df.columns and 'primaryspaceusage' in df.columns:
                    for (site, usage), group in df.groupby(['site_id', 'primaryspaceusage']):
                        group_median = group[col].median()
                        if pd.notna(group_median):
                            mask = (df['site_id'] == site) & (df['primaryspaceusage'] == usage) & df[col].isna()
                            df.loc[mask, col] = group_median
                
                # Fallback to global median for remaining NaNs
                if df[col].isna().any():
                    global_median = df[col].median()
                    df[col] = df[col].fillna(global_median if pd.notna(global_median) else 0)
        
        # Handle hierarchical categorical: sub_primaryspaceusage from primaryspaceusage
        if 'sub_primaryspaceusage' in df.columns and 'primaryspaceusage' in df.columns:
            if df['sub_primaryspaceusage'].isna().any():
                # For each primaryspaceusage, find most common sub category
                for primary in df['primaryspaceusage'].unique():
                    mask = (df['primaryspaceusage'] == primary)
                    mode_sub = df.loc[mask, 'sub_primaryspaceusage'].mode()
                    if len(mode_sub) > 0:
                        df.loc[mask & df['sub_primaryspaceusage'].isna(), 'sub_primaryspaceusage'] = mode_sub[0]
        
        # Handle remaining categorical columns
        categorical_cols = [col for col in self.cfg.context_vars.keys() 
                          if col not in numeric_cols and col in df.columns]
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    # Create 'unknown' category instead of dropping
                    df[col] = df[col].fillna('unknown')
        
        return df
    
    def _reduce_high_cardinality_features(self, df, col, min_samples=50, max_categories=30):
        """
        Reduce high-cardinality categorical features by grouping rare categories.
        
        Strategy: Keep top N categories by frequency, group the rest as 'other_{parent}'
        where parent is the primaryspaceusage.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            col (str): Column name to reduce
            min_samples (int): Minimum samples to keep category separate
            max_categories (int): Maximum number of categories to keep
        
        Returns:
            pd.DataFrame: DataFrame with reduced categories
        """
        if col not in df.columns:
            return df
        
        df = df.copy()
        value_counts = df[col].value_counts()
        
        # Keep categories with enough samples
        keep_categories = value_counts[value_counts >= min_samples].index.tolist()
        
        # If still too many, keep only top max_categories
        if len(keep_categories) > max_categories:
            keep_categories = value_counts.nlargest(max_categories).index.tolist()
        
        # For sub_primaryspaceusage, group by parent category
        if col == 'sub_primaryspaceusage' and 'primaryspaceusage' in df.columns:
            # Map rare subcategories to 'other_{primary}'
            def map_category(row):
                if row[col] in keep_categories:
                    return row[col]
                else:
                    return f"other_{row['primaryspaceusage']}"
            df[col] = df.apply(map_category, axis=1)
        else:
            # Generic grouping
            df.loc[~df[col].isin(keep_categories), col] = 'other'
        
        return df

        