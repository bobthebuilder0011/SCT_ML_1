# House Price Prediction - Enhanced

A robust and modular machine learning project that predicts house prices using Linear Regression.

## Enhancements
- **Modular Architecture**: Core logic extracted to `model.py` for reuse between training and production.
- **Model Persistence**: Models and scalers are now saved to disk (`house_model.joblib`), avoiding redundant training.
- **Automated Testing**: Unit tests included to ensure data preprocessing and model logic are correct.
- **Improved UI**: Streamlit application with better error handling, input validation, and caching.
- **Robust Preprocessing**: Handles missing values and outliers specifically tuned for housing data.

## Project Structure

```
.
├── model.py                        # Centralized HousePriceModel class
├── house_price_linear_regression.py # Script to train, evaluate & save model
├── app.py                           # Interactive Streamlit web application
├── test_model.py                   # Automated unit tests
├── requirements.txt                 # Updated Python dependencies
└── train.csv                       # Dataset
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure `train.csv` is in the project root.

## Usage

### 1. Train the Model
Train the model and generate performance visualizations:
```bash
python house_price_linear_regression.py
```
This will save `house_model.joblib` and `scaler.joblib`.

### 2. Run Tests
Ensure everything is working correctly:
```bash
pytest test_model.py
```

### 3. Launch the Web App
```bash
streamlit run app.py
```

## Dataset Requirements
The model expects a CSV file named `train.csv` with the following columns:
- `PROPERTYSQFT`: Square footage
- `BEDS`: Number of bedrooms
- `BATH`: Number of bathrooms
- `PRICE`: Target price

## Model Details
- **Algorithm**: Linear Regression
- **Scaler**: StandardScaler (persisted)
- **Features**: Square Footage, Bedrooms, Bathrooms
- **Validation**: 80/20 train-test split with R², RMSE, and MAE metrics.

## Repository Updates
To update your GitHub repository with these changes, use:
```bash
git add .
git commit -m "Refactor: Modularize HousePriceModel, add unit tests and UI improvements"
git push origin main
```
