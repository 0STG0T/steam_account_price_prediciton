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
from train_columns import train_columns


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
        # Ensure columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_columns}")
        
        # Perform one-hot encoding
        encoded_df = pd.get_dummies(df, columns=columns, drop_first=False)
        
        return encoded_df
        
    def load_model(self, fname: str):
        """
        Loads a pre-trained CatBoost model from a file.

        Parameters:
        - fname: str - Path to the model file.
        """
        self.meta_model = CatBoostRegressor().load_model(fname=fname, format='onnx')
        print("Loaded the model from", fname)
    
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

        # Drop unnecessary columns
        df = df.drop(columns=['steam_cards_count', 'steam_cards_games', 'category_id', 'is_sticky'])

        # Convert timestamp columns to datetime
        date_cols = ['published_date', 'update_stat_date', 'refreshed_date', 'steam_register_date',
                     'steam_last_activity', 'steam_cs2_last_activity', 'steam_cs2_ban_date',
                     'steam_last_transaction_date', 'steam_market_ban_end_date', 'steam_cs2_last_launched']
        for col in date_cols:
            df[col] = df[col].apply(lambda x: datetime.fromtimestamp(x) if x != 0 else np.NaN)

        # Extract time features
        for col in date_cols:
            df = self.extract_time_features(df, col)
        df = df.drop(columns=date_cols)

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
        
        # Ensure all required columns are present
        if len(set(train_columns) - set(df.columns)) > 0:
            for c in list(set(train_columns) - set(df.columns)):
                df[c] = 0
            
        df = df[train_columns]
        
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

    def train(self, df):
        """
        Trains the CatBoost model on the provided dataset.

        Parameters:
        - df: pd.DataFrame - Training dataset.

        Returns:
        - CatBoostRegressor - Trained model.
        """
        df = self.preprocess_data(df)
        X_train = df.drop(columns=['sold_price'])
        y_train = df['sold_price']
        
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

        self.meta_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=250)
        
        print("Training complete.")
        
        return self.meta_model

    def validate(self, valid_df, save_plot_path="pearson_vs_samples.png"):
        """
        Validates the model on a validation set.

        Parameters:
        - valid_df: pd.DataFrame - DataFrame of base structure on which we validate the model.
        - save_plot_path: str - Path to save the Pearson correlation plot.

        Returns:
        - dict - Regression metrics and Pearson correlations.
        """
        # Preprocess validation data
        valid_df = self.preprocess_data(valid_df)
        X_val = valid_df.drop(columns=['sold_price'])
        y_val = valid_df['sold_price']
        
        # Predict using the model
        preds = self.meta_model.predict(X_val)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        pearson_corr_full = self.pearson_correlation_preds_yval(preds, y_val)
        
        # Pearson correlations for subsets
        sample_pearsons = {}
        for size in [100, 1000, 10000]:
            if len(y_val) >= size:
                indices = np.random.choice(len(y_val), size=size, replace=False)
                sampled_preds = preds[indices]
                sampled_y_val = y_val.iloc[indices]
                sample_pearsons[size] = self.pearson_correlation_preds_yval(sampled_preds, sampled_y_val)
            else:
                sample_pearsons[size] = None
        
        # Print and log metrics
        print("\nRegression Metrics:\n")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Pearson Correlation (Full Dataset): {pearson_corr_full:.4f}")
        for size, corr in sample_pearsons.items():
            if corr is not None:
                print(f"Pearson Correlation (Sample Size {size}): {corr:.4f}")
            else:
                print(f"Pearson Correlation (Sample Size {size}): Not enough rows in validation set")
        
        # Generate Pearson correlation vs. number of samples plot
        self._plot_pearson_correlation(preds, y_val, save_path=save_plot_path)
        
        # Return metrics as a dictionary
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "pearson_correlation_full": pearson_corr_full,
            "sample_pearsons": sample_pearsons
        }
        
    def finetune(self, df, use_crossval=True):
        """
        Fine-tunes the existing CatBoost model with new data.

        Args:
            df (DataFrame): DataFrame containing the new data to fine-tune on.
            use_crossval (bool, optional): Whether to use cross-validation during fine-tuning. Defaults to True.
        """
        # Preprocess the new data
        df = self.preprocess_data(df)
        
        # Separate features and target variable
        X_train = df.drop(columns=['sold_price'])
        y_train = df['sold_price']
        
        # Initialize a new CatBoostRegressor with the same parameters as the meta_model
        new_model = CatBoostRegressor(
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
        
        # Check if meta_model is available and is of the correct type
        if not hasattr(self, 'meta_model') or not isinstance(self.meta_model, CatBoostRegressor):
            raise ValueError("meta_model is not available or not a CatBoostRegressor instance.")
        
        try:
            if use_crossval:
                # Split the data into training and validation sets
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, random_state=42)
                
                # Fine-tune the model with cross-validation
                new_model.fit(X_train_split, y_train_split, 
                            eval_set=(X_test_split, y_test_split), 
                            verbose=0, 
                            init_model=self.meta_model)
            else:
                # Fine-tune the model without cross-validation
                new_model.fit(X_train, y_train, 
                            verbose=0, 
                            init_model=self.meta_model)
            
            # Update the meta_model with the fine-tuned model
            self.meta_model = new_model
            print("Finetuning complete.")
        except Exception as e:
            print(f"An error occurred during finetuning: {e}")

    @staticmethod
    def pearson_correlation_preds_yval(preds, y_val):
        """
        Calculates the Pearson correlation coefficient between predicted and true values.

        Parameters:
        - preds: np.array - Predicted values.
        - y_val: np.array - True values.

        Returns:
        - float - Pearson correlation coefficient.
        """
        preds, y_val = np.array(preds), np.array(y_val)
        return np.corrcoef(preds, y_val)[0, 1]

    def export(self, output_path_onnx):
        """
        Exports the trained model to an ONNX file.

        Parameters:
        - output_path: str - Path to save the ONNX model.
        """
        self.meta_model.save_model(
            fname=output_path_onnx,
            format="onnx",
            export_parameters={
                'onnx_domain': 'ai.catboost',
                'onnx_model_version': 1,
                'onnx_doc_string': f'Model for category {self.category_number}',
                'onnx_graph_name': f'Category {self.category_number} CatBoost Regressor'
            }
        )

        print(f"Model exported to {output_path_onnx}")

    def _plot_pearson_correlation(self, preds, y_val, save_path="pearson_vs_samples.png"):
        """
        Generates a plot for Pearson correlation vs. number of samples with trend and LOWESS.

        Parameters:
        - preds: np.array - Predicted values.
        - y_val: np.array - True values.
        - save_path: str - Path to save the plot.
        """
        # Prepare data for plotting
        sample_sizes = np.linspace(10, len(preds), 100, dtype=int)
        correlations = []

        for size in sample_sizes:
            indices = np.random.choice(len(preds), size=size, replace=False)
            sampled_preds = preds[indices]
            sampled_y_val = y_val.iloc[indices]
            correlations.append(self.pearson_correlation_preds_yval(sampled_preds, sampled_y_val))
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            "Sample Size": sample_sizes,
            "Pearson Correlation": correlations
        })
        
        # Calculate LOWESS trend
        lowess_result = lowess(plot_df["Pearson Correlation"], plot_df["Sample Size"], frac=0.3)
        lowess_x, lowess_y = zip(*lowess_result)
        
        # Plot with seaborn
        sns.set(style="whitegrid", context="talk")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Sample Size", y="Pearson Correlation", data=plot_df, s=30, alpha=0.7, color="blue", label="Correlation")
        sns.lineplot(x="Sample Size", y="Pearson Correlation", data=plot_df, color="orange", linewidth=2, label="Trend")
        plt.plot(lowess_x, lowess_y, color="red", linestyle="--", linewidth=2, label="LOWESS Curve")
        
        # Add labels and title
        plt.title(f"Pearson Correlation vs. Sample Size (Category {self.category_number})", fontsize=16)
        plt.xlabel("Sample Size", fontsize=14)
        plt.ylabel("Pearson Correlation Coefficient", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path)
        plt.close()
        print(f"Pearson correlation plot saved to {save_path}")