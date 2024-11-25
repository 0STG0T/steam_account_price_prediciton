import pandas as pd
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor
import re
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from ..data_processing.train_columns import train_columns


warnings.filterwarnings('ignore')

class SingleCategoryModel:
    def __init__(self, category_number):
        """
        Initializes the SingleCategoryModel class with the category number.

        Parameters:
        - category_number: int - The category number for which the model is trained.
        """
        self.category_number = category_number
        self.meta_model = None
        print(f"Initialized SingleCategoryModel for category {self.category_number}.")
        
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
        """
        Loads a pre-trained CatBoost model from a file.

        Parameters:
        - fname: str - Path to the model file.
        - format: str - Format of the model file ('auto', 'cbm', or 'onnx'). Default is 'auto'.
        """
        if format == 'auto':
            format = 'onnx' if fname.endswith('.onnx') else 'cbm'
        self.meta_model = CatBoostRegressor().load_model(fname=fname, format=format)
        print(f"Loaded the model from {fname} in {format} format")

    @classmethod
    def load(cls, model_path):
        """Load a trained model from file."""
        model = cls(category_number=1)  # Default category, will be overwritten
        if model_path.endswith('.onnx'):
            # Save native format alongside ONNX for inference
            native_path = model_path.replace('.onnx', '.cbm')
            model.meta_model = CatBoostRegressor()
            model.meta_model.load_model(native_path)
        else:
            model.meta_model = CatBoostRegressor()
            model.meta_model.load_model(model_path)
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

    def train(self, df, callback=None):
        """
        Trains the CatBoost model on the provided dataset.

        Parameters:
        - df: pd.DataFrame - Training dataset.
        - callback: callable - Optional callback function that receives (epoch, metrics) as parameters
               for logging training progress.

        Returns:
        - CatBoostRegressor - Trained model.
        """
        df = self.preprocess_data(df)
        target_col = 'target' if 'target' in df.columns else 'sold_price'
        X_train = df.drop(columns=[target_col])
        y_train = df[target_col]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, shuffle=True)

        self.meta_model = CatBoostRegressor(
            iterations=60000,
            l2_leaf_reg=2.7,
            use_best_model=True,
            early_stopping_rounds=300,
            posterior_sampling=True,
            grow_policy='SymmetricTree',
            bootstrap_type='Bernoulli',
            random_state=42,
            leaf_estimation_method='Newton',
            score_function='Cosine',
            colsample_bylevel=0.94,
            thread_count=4,
        )

        class MetricsCallback:
            def __init__(self, user_callback):
                self.user_callback = user_callback

            def after_iteration(self, info):
                if self.user_callback and info.iteration % 250 == 0:
                    metrics = {
                        'iteration': info.iteration,
                        'learn_loss': info.metrics['learn']['RMSE'],
                        'validation_loss': info.metrics['validation']['RMSE']
                    }
                    self.user_callback(info.iteration, metrics)
                return True

        callbacks = [MetricsCallback(callback)] if callback else None

        self.meta_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=250,
            callbacks=callbacks
        )

        print("Training complete.")

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

    def validate(self, df, save_plot_path="pearson_vs_samples.png"):
        """
        Validates the model performance on a dataset.

        Parameters:
        - df: pd.DataFrame - Validation dataset
        - save_plot_path: str - Path to save correlation plot

        Returns:
        - dict - Dictionary containing validation metrics
        """
        if self.meta_model is None:
            raise ValueError("Model must be trained before validation")

        df = self.preprocess_data(df)
        target_col = 'target' if 'target' in df.columns else 'sold_price'
        X_val = df.drop(columns=[target_col])
        y_val = df[target_col]

        preds = self.meta_model.predict(X_val)

        metrics = {
            'mae': mean_absolute_error(y_val, preds),
            'mse': mean_squared_error(y_val, preds),
            'rmse': np.sqrt(mean_squared_error(y_val, preds)),
            'r2': r2_score(y_val, preds)
        }

        print("\nRegression Metrics:\n")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")

        # Calculate Pearson correlations for different sample sizes
        metrics['pearson_full'] = np.corrcoef(preds, y_val)[0, 1]
        print(f"Pearson Correlation (Full Dataset): {metrics['pearson_full']:.4f}")

        if save_plot_path:
            self._plot_pearson_correlation(preds, y_val, save_path=save_plot_path)

        return metrics

    def finetune(self, df, epochs=5):
        """
        Finetunes the model on new data.

        Parameters:
        - df: pd.DataFrame - New training data
        - epochs: int - Number of epochs for finetuning
        """
        if not hasattr(self, 'meta_model') or not isinstance(self.meta_model, CatBoostRegressor):
            raise ValueError("meta_model is not available or not a CatBoostRegressor instance.")

        df = self.preprocess_data(df)
        target_col = 'target' if 'target' in df.columns else 'sold_price'
        X_train = df.drop(columns=[target_col])
        y_train = df[target_col]

        try:
            self.meta_model.fit(
                X_train, y_train,
                init_model=self.meta_model,
                verbose=False
            )
            print("Finetuning complete.")
        except Exception as e:
            print(f"An error occurred during finetuning: {e}")

    @staticmethod
    def pearson_correlation_preds_yval(preds, y_val):
        """Calculate Pearson correlation between predictions and actual values."""
        preds, y_val = np.array(preds), np.array(y_val)
        return np.corrcoef(preds, y_val)[0, 1]

    def export(self, model_path):
        """Export model to file."""
        if model_path.endswith('.onnx'):
            # Save native format first
            native_path = model_path.replace('.onnx', '.cbm')
            self.meta_model.save_model(native_path)
            # Then export to ONNX
            self.meta_model.save_model(model_path, format="onnx")
            print(f"Model exported to {model_path} and {native_path}")
        else:
            self.meta_model.save_model(model_path)
            print(f"Model exported to {model_path}")
        return model_path

    def _plot_pearson_correlation(self, preds, y_val, save_path="pearson_vs_samples.png"):
        """
        Generates a plot for Pearson correlation vs. number of samples with LOWESS curve.

        Parameters:
        - preds: np.array - Predicted values
        - y_val: np.array - True values
        - save_path: str - Path to save the plot
        """
        try:
            # Calculate correlations for different sample sizes
            sample_sizes = np.linspace(100, len(preds), 20, dtype=int)
            correlations = []

            for size in sample_sizes:
                indices = np.random.choice(len(preds), size=size, replace=False)
                corr = self.pearson_correlation_preds_yval(preds[indices], y_val[indices])
                correlations.append(corr)

            # Create figure
            plt.figure(figsize=(12, 6))

            # Plot scatter of correlations vs sample sizes
            plt.scatter(sample_sizes, correlations, alpha=0.5, label='Sample Correlations')

            # Add LOWESS curve
            smoothed = lowess(correlations, sample_sizes, frac=0.6)
            plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', label='LOWESS Curve')

            # Add full dataset correlation
            full_corr = self.pearson_correlation_preds_yval(preds, y_val)
            plt.axhline(y=full_corr, color='g', linestyle='--',
                       label=f'Full Dataset Correlation ({full_corr:.3f})')

            plt.xlabel('Number of Samples')
            plt.ylabel('Pearson Correlation')
            plt.title('Pearson Correlation vs. Number of Samples')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path)
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate correlation plot: {str(e)}")
