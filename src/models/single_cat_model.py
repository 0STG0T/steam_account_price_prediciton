import os
import pandas as pd
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor
import re
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processing.train_columns import train_columns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
from scipy.stats import norm
import warnings
from src.models.custom_metric import UserDefinedMetric, UserDefinedObjective

warnings.filterwarnings('ignore')

class SingleCategoryModel:
    def __init__(self, category_number, params=None):
        """Initialize SingleCategoryModel."""
        self.category_number = category_number
        base_params = {
            'grow_policy': 'SymmetricTree',
            'bootstrap_type': 'Bernoulli',
            #'eval_metric': UserDefinedMetric(penalty_factor=2.0),
            'eval_metric': 'RMSE',
            'loss_function': UserDefinedObjective(penalty_factor=1.2),
        }
        if params:
            base_params.update(params)
        self.meta_model = CatBoostRegressor(**base_params)

    def one_hot_encode_and_drop(self, df, columns):
        """
        One-hot-encodes the specified columns in a DataFrame and drops the original columns.

        Parameters:
        - df: pd.DataFrame - Input DataFrame.
        - columns: list of str - List of column names to be one-hot-encoded.

        Returns:
        - pd.DataFrame - Updated DataFrame with one-hot-encoded columns and original columns dropped.
        """
        # Add missing columns with default value
        for col in columns:
            if col not in df.columns:
                df[col] = 'unknown'

        # Perform one-hot encoding
        encoded_df = pd.get_dummies(df, columns=columns, drop_first=False)

        return encoded_df

    def load_model(self, fname: str, format='auto'):
        """Load a pre-trained CatBoost model."""
        if format == 'auto':
            format = 'onnx'
        self.meta_model = CatBoostRegressor().load_model(fname=fname, format=format)

    @classmethod
    def load(cls, model_path):
        """Load a trained model from file."""
        model = cls(category_number=1)
        base_params = {
            'grow_policy': 'SymmetricTree',
            'bootstrap_type': 'Bernoulli',
            'eval_metric': 'RMSE',
            'loss_function': UserDefinedObjective(penalty_factor=1.2),
        }
        model.meta_model = CatBoostRegressor(**base_params)
        model.meta_model.load_model(model_path, format='onnx')
        return model

    def sum_playtime(self, x):
        """
        Sums the playtime from a nested dictionary structure.

        Parameters:
        - x: dict - Nested dictionary containing playtime data.

        Returns:
        - float - Sum of playtime.
        """
        s = 0
        if isinstance(x, dict) and 'list' in x:
            for key in x['list']:
                s += float(x['list'][key]['playtime_forever'])
        return s

    def std_playtime(self, x):
        """
        Calculates the standard deviation of playtime from a nested dictionary structure.

        Parameters:
        - x: dict - Nested dictionary containing playtime data.

        Returns:
        - float - Standard deviation of playtime.
        """
        playtimes = []
        if isinstance(x, dict) and 'list' in x:
            for key in x['list']:
                playtimes.append(float(x['list'][key]['playtime_forever']))
            return np.std(playtimes)
        return 0

    def preprocess_data(self, df):
        """
        Preprocesses the input dataset.

        Parameters:
        - df: DataFrame.

        Returns:
        - pd.DataFrame - Preprocessed DataFrame.
        """
        df = df.copy()
        target_col = 'target' if 'target' in df.columns else 'sold_price'
        target_values = df[target_col].copy() if target_col in df.columns else None

        # Drop columns if they exist
        columns_to_drop = ['steam_cards_count', 'steam_cards_games', 'category_id', 'is_sticky']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df = df.drop(columns=existing_columns)

        # Convert timestamp columns to datetime if they exist
        date_cols = ['published_date', 'update_stat_date', 'refreshed_date', 'steam_register_date',
                     'steam_last_activity', 'steam_cs2_last_activity', 'steam_cs2_ban_date',
                     'steam_last_transaction_date', 'steam_market_ban_end_date', 'steam_cs2_last_launched']
        existing_date_cols = [col for col in date_cols if col in df.columns]
        for col in existing_date_cols:
            df[col] = df[col].apply(lambda x: datetime.fromtimestamp(x) if x != 0 else np.NaN)

        # Extract time features for existing columns
        for col in existing_date_cols:
            df = self.extract_time_features(df, col)
        if existing_date_cols:
            df = df.drop(columns=existing_date_cols)

        # Handle `steam_balance`
        df['steam_currency'] = df['steam_balance'].apply(lambda x: self.remove_numbers_dots_dashes(x))
        df = df.drop(columns=['steam_balance'])

        # Sum columns
        df['inv_value_sum'] = df.filter(like='inv_value').sum(axis=1)
        df['game_count_sum'] = df.filter(like='game_count').sum(axis=1)
        df['level_sum'] = df.filter(like='level').sum(axis=1)

        # Additional feature engineering
        df['price_per_view'] = df['price'] / df['view_count']

        # steam_full_games handling
        df['total_steam_games'] = df['steam_full_games'].apply(lambda x: x['total'] if 'total' in x else -1)
        df['total_playtime'] = df['steam_full_games'].apply(lambda x: self.sum_playtime(x))
        df['std_playtime'] = df['steam_full_games'].apply(lambda x: self.std_playtime(x))

        df = df.drop(columns=['steam_full_games'])

        # One-hot encode categorical features
        cat_features = ['item_origin', 'extended_guarantee', 'nsb', 'email_type', 'item_domain',
                        'resale_item_origin', 'steam_country', 'steam_community_ban', 'steam_is_limited',
                        'steam_cs2_wingman_rank_id', 'steam_cs2_rank_id', 'steam_cs2_ban_type', 'steam_currency'] + \
                       [col for col in df.columns if 'is_weekend' in col]

        df = self.one_hot_encode_and_drop(df=df, columns=cat_features)

        # Add missing columns efficiently
        missing_cols = list(set(train_columns) - set(df.columns))
        if missing_cols:
            missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, missing_df], axis=1)

        df = df[train_columns]

        if target_values is not None:
            df[target_col] = target_values

        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def extract_time_features(df, col):
        """
        Extracts various time-related features from a datetime column.

        Parameters:
        - df: pd.DataFrame - Input DataFrame.
        - col: str - Name of the datetime column.

        Returns:
        - pd.DataFrame - DataFrame with added time features.
        """
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month
        df[col + '_day'] = df[col].dt.day
        df[col + '_hour'] = df[col].dt.hour
        df[col + '_minute'] = df[col].dt.minute
        df[col + '_second'] = df[col].dt.second
        df[col + '_weekday'] = df[col].dt.weekday
        df[col + '_is_weekend'] = df[col].dt.weekday.isin([5, 6]).astype(int)
        return df

    @staticmethod
    def remove_numbers_dots_dashes(s):
        """
        Removes numbers, dots, and dashes from a string.

        Parameters:
        - s: str - Input string.

        Returns:
        - str - Cleaned string.
        """
        return re.sub(r'[0-9.,-]', '', s) if isinstance(s, str) else s

    def train(self, df, callback=None, is_finetuning=False, **kwargs):
        """Train or fine-tune the CatBoost model."""
        df = self.preprocess_data(df)
        target_col = 'target' if 'target' in df.columns else 'sold_price'
        X_train = df.drop(columns=[target_col])
        y_train = df[target_col].astype(float)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, shuffle=True)

        if is_finetuning:
            finetune_params = {
                'grow_policy': 'SymmetricTree',
                'bootstrap_type': 'Bernoulli',
                'iterations': kwargs.get('iterations', 1000),
                'learning_rate': kwargs.get('learning_rate', 0.03),
                'early_stopping_rounds': 50,
                'use_best_model': True
            }
            self.meta_model = CatBoostRegressor(**finetune_params)

        fit_params = {
            'X': X_train,
            'y': y_train,
            'eval_set': [(X_val, y_val)],
            'verbose': 250
        }

        self.meta_model.fit(**fit_params)
        return self.meta_model

    def predict(self, df):
        """
        Make predictions on the input DataFrame.

        Parameters:
        - df: pd.DataFrame - Input data for prediction.

        Returns:
        - np.ndarray - Array of predictions.
        """
        df = self.preprocess_data(df)
        return self.meta_model.predict(df)

    def predict_single(self, sample):
        """
        Make prediction on a single sample.

        Parameters:
        - sample: dict or pd.Series - Single sample of input data.

        Returns:
        - float - Predicted value.
        """
        if isinstance(sample, dict):
            df = pd.DataFrame([sample])
        else:
            df = sample.to_frame().T

        df = self.preprocess_data(df)
        return float(self.meta_model.predict(df[train_columns])[0])

    def validate(self, df, save_dir):
        """Validate model performance and save improved, informative plots."""
        if self.meta_model is None:
            raise ValueError("Model must be trained before validation")

        # Reset index and preprocess data
        df = df.reset_index(drop=False)
        df = self.preprocess_data(df)
        
        # Determine target column
        target_col = 'target' if 'target' in df.columns else 'sold_price'
        X_val = df.drop(columns=[target_col])
        y_val = df[target_col]

        # Generate predictions
        preds = self.meta_model.predict(X_val)

        # Compute metrics
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        pearson_corr = np.corrcoef(preds, y_val)[0, 1]
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'pearson': pearson_corr
        }

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Sample data for scatter plots
        sample_size = min(500, len(y_val))  # Limit to 500 points for visualization
        sampled_indices = np.random.choice(len(y_val), size=sample_size, replace=False)
        y_val_sampled = y_val.iloc[sampled_indices]
        preds_sampled = preds[sampled_indices]

        # Create Pearson correlation data starting from 1000 samples
        sample_sizes = np.arange(1000, len(y_val) + 1, 100)  # Start at 1000 samples
        pearson_values = [np.corrcoef(preds[:n], y_val[:n])[0, 1] for n in sample_sizes]

        # Apply LOWESS smoothing
        lowess_curve = lowess(pearson_values, sample_sizes, frac=0.2)

        # Create plots
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        sns.set_theme(style="whitegrid")

        # 1. Pearson Correlation vs Number of Samples with LOWESS curve
        sns.lineplot(x=sample_sizes, y=pearson_values, ax=axs[0, 0], label="Pearson Correlation", marker="o")
        axs[0, 0].plot(lowess_curve[:, 0], lowess_curve[:, 1], color="orange", linestyle="--", label="LOWESS Curve")
        axs[0, 0].fill_between(lowess_curve[:, 0], lowess_curve[:, 1] - 0.01, lowess_curve[:, 1] + 0.01, color="orange", alpha=0.2)
        axs[0, 0].axhline(0.95, color='green', linestyle=":", label="Threshold")
        axs[0, 0].set_title("Pearson Correlation vs Number of Samples (Start: 1000)", fontsize=16)
        axs[0, 0].set_xlabel("Number of Samples", fontsize=14)
        axs[0, 0].set_ylabel("Pearson Correlation", fontsize=14)
        axs[0, 0].legend(fontsize=12)
        axs[0, 0].tick_params(axis='both', labelsize=12)

        # 2. Predicted vs Actual Prices with density
        scatter = axs[0, 1].scatter(y_val_sampled, preds_sampled, c=np.abs(y_val_sampled - preds_sampled), cmap="coolwarm", alpha=0.7, edgecolor="k")
        axs[0, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color="red", linestyle="--", label="Ideal Fit")
        axs[0, 1].set_title("Predicted vs Actual Prices", fontsize=16)
        axs[0, 1].set_xlabel("Actual Prices", fontsize=14)
        axs[0, 1].set_ylabel("Predicted Prices", fontsize=14)
        fig.colorbar(scatter, ax=axs[0, 1], label="Absolute Residuals")
        axs[0, 1].legend(fontsize=12)

        # 3. Residuals vs Predicted Prices
        residuals = y_val_sampled - preds_sampled
        sns.scatterplot(x=preds_sampled, y=residuals, ax=axs[1, 0], alpha=0.7, hue=residuals, palette="coolwarm", edgecolor="k")
        sns.lineplot(x=preds_sampled, y=[0]*len(preds_sampled), ax=axs[1, 0], color='red', linestyle="--")
        axs[1, 0].set_title("Residuals vs Predicted Prices", fontsize=16)
        axs[1, 0].set_xlabel("Predicted Prices", fontsize=14)
        axs[1, 0].set_ylabel("Residuals", fontsize=14)
        axs[1, 0].legend(title="Residuals", fontsize=12)

        # 4. Histogram of Residuals with Gaussian fit
        sns.histplot(residuals, bins=30, kde=True, ax=axs[1, 1], color="orange", edgecolor="k", alpha=0.8)
        mu, std = np.mean(residuals), np.std(residuals)
        x = np.linspace(mu - 3*std, mu + 3*std, 100)
        axs[1, 1].plot(x, norm.pdf(x, mu, std) * len(residuals) * np.diff(np.histogram(residuals, bins=30)[1])[0], color="blue", linestyle="--", label="Gaussian Fit")
        axs[1, 1].set_title("Distribution of Residuals", fontsize=16)
        axs[1, 1].set_xlabel("Residuals", fontsize=14)
        axs[1, 1].set_ylabel("Frequency", fontsize=14)
        axs[1, 1].legend(fontsize=12)

        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "improved_validation_plots.png")
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

        print(f"Improved validation plots saved to {plot_path}")

        return metrics

    @staticmethod
    def pearson_correlation_preds_yval(preds, y_val):
        """Calculate Pearson correlation between predictions and actual values."""
        preds, y_val = np.array(preds), np.array(y_val)
        return np.corrcoef(preds, y_val)[0, 1]

    def export(self, model_path):
        """Export model to ONNX format."""
        # Remove any existing extensions
        base_path = os.path.splitext(model_path)[0]
        if base_path.endswith('.cbm'):
            base_path = base_path[:-4]
        onnx_path = f"{base_path}.onnx"
        self.meta_model.save_model(onnx_path, format="onnx")
        print(f'\nModel saved to {onnx_path}\n')
        return onnx_path
