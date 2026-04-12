import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

class HousePriceModel:
    def __init__(self, model_path='house_model.joblib', scaler_path='scaler.joblib'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = ['SquareFootage', 'Bedrooms', 'Bathrooms']
        self.target_name = 'SalePrice'

    def load_raw_data(self, file_path='train.csv'):
        """Loads and prepares the raw data from CSV."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_cols = ['PROPERTYSQFT', 'BEDS', 'BATH', 'PRICE']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        # Map to internal names
        data = pd.DataFrame({
            'SquareFootage': df['PROPERTYSQFT'],
            'Bedrooms': df['BEDS'],
            'Bathrooms': df['BATH'],
            'SalePrice': df['PRICE']
        })
        return data

    def preprocess(self, data, is_training=True):
        """Cleans data and handles outliers."""
        # 1. Handle Missing Values
        data = data.dropna()
        
        # 2. Outlier Removal (Adjusted for NY Housing prices)
        if is_training:
            data = data[
                (data['SquareFootage'] < 8000) & 
                (data['SalePrice'] < 4000000)
            ]
        
        return data

    def train(self, data):
        """Trains the model and returns performance metrics."""
        X = data[self.feature_names]
        y = data[self.target_name]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Model Training
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        return metrics

    def save(self):
        """Saves the model and scaler to disk."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not trained yet.")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load(self):
        """Loads the model and scaler from disk."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            return False
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        return True

    def predict(self, sqft, beds, baths):
        """Makes a prediction for a single house."""
        if self.model is None or self.scaler is None:
            if not self.load():
                raise ValueError("Model not trained or loaded.")
        
        # Create a DataFrame with feature names to avoid warnings and ensure consistency
        input_df = pd.DataFrame([[sqft, beds, baths]], columns=self.feature_names)
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)[0]
        return prediction
