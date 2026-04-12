import pytest
import pandas as pd
import os
import numpy as np
from model import HousePriceModel

def test_model_initialization():
    hpm = HousePriceModel()
    assert hpm.model is None
    assert hpm.scaler is None
    assert len(hpm.feature_names) == 3

def test_load_data_error():
    hpm = HousePriceModel()
    with pytest.raises(FileNotFoundError):
        hpm.load_raw_data("non_existent.csv")

def test_preprocess_missing_values():
    hpm = HousePriceModel()
    df = pd.DataFrame({
        'SquareFootage': [1000, 2000, np.nan],
        'Bedrooms': [2, 3, 4],
        'Bathrooms': [1, 2, 2.5],
        'SalePrice': [100000, 200000, 300000]
    })
    processed = hpm.preprocess(df)
    assert len(processed) == 2
    assert not processed.isnull().values.any()

def test_preprocess_outliers():
    hpm = HousePriceModel()
    df = pd.DataFrame({
        'SquareFootage': [1000, 10000],  # Outlier
        'Bedrooms': [2, 3],
        'Bathrooms': [1, 2],
        'SalePrice': [100000, 5000000] # Outlier
    })
    processed = hpm.preprocess(df, is_training=True)
    assert len(processed) == 1
    assert processed.iloc[0]['SquareFootage'] == 1000

def test_train_and_predict():
    hpm = HousePriceModel()
    # Create dummy training data
    df = pd.DataFrame({
        'SquareFootage': [1000, 2000, 1500, 3000, 2500, 1200],
        'Bedrooms': [2, 3, 3, 4, 3, 2],
        'Bathrooms': [1, 2, 2, 3, 2, 1],
        'SalePrice': [200000, 400000, 300000, 600000, 500000, 240000]
    })
    
    metrics = hpm.train(df)
    assert 'r2' in metrics
    assert hpm.model is not None
    assert hpm.scaler is not None
    
    # Test prediction
    prediction = hpm.predict(1500, 3, 2)
    assert isinstance(prediction, float)
    assert prediction > 0

def test_save_load():
    hpm = HousePriceModel(model_path='test_model.joblib', scaler_path='test_scaler.joblib')
    df = pd.DataFrame({
        'SquareFootage': [1000, 2000, 1500, 3000, 2500, 1200],
        'Bedrooms': [2, 3, 3, 4, 3, 2],
        'Bathrooms': [1, 2, 2, 3, 2, 1],
        'SalePrice': [200000, 400000, 300000, 600000, 500000, 240000]
    })
    hpm.train(df)
    hpm.save()
    
    hpm2 = HousePriceModel(model_path='test_model.joblib', scaler_path='test_scaler.joblib')
    assert hpm2.load() is True
    assert hpm2.predict(1500, 3, 2) == hpm.predict(1500, 3, 2)
    
    # Cleanup
    os.remove('test_model.joblib')
    os.remove('test_scaler.joblib')
