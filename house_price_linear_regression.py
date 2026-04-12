import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import HousePriceModel

def train_and_save():
    print("="*60)
    print("      HOUSE PRICE PREDICTION: LINEAR REGRESSION")
    print("="*60)

    model = HousePriceModel()
    
    # 1. Load Data
    try:
        df_raw = model.load_raw_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Preprocess
    data = model.preprocess(df_raw)
    print(f"🧹 Data Cleaning: Processed {len(data)} rows.")

    # 3. Train
    metrics = model.train(data)
    
    # 4. Save
    model.save()
    print("💾 Model and scaler saved successfully.")

    # 5. Output Evaluation
    print("\n--- Model Performance ---")
    print(f"R-squared (Accuracy) Score: {metrics['r2']:.4f}")
    print(f"Root Mean Squared Error: ${metrics['rmse']:,.2f}")
    print(f"Mean Absolute Error: ${metrics['mae']:,.2f}")

    # 6. Feature Impact (Coefficients)
    print("\n--- Feature Impact (Coefficients) ---")
    for name, coef in zip(model.feature_names, model.model.coef_):
        print(f"{name:15}: +${coef:,.2f} impact per unit change")

    # 7. Visualizations
    # We'll re-run a small prediction for plotting
    X = data[model.feature_names]
    y = data[model.target_name]
    X_scaled = model.scaler.transform(X)
    y_pred = model.model.predict(X_scaled)

    plt.figure(figsize=(12, 5))
    
    # Plot 1: Prediction Accuracy
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y, y=y_pred, alpha=0.5, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', lw=2, label='Ideal Fit')
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.legend()

    # Plot 2: Error Distribution
    plt.subplot(1, 2, 2)
    residuals = y - y_pred
    sns.histplot(residuals, kde=True, color='teal')
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error Amount ($)")

    plt.tight_layout()
    plt.savefig('performance_plots.png')
    print("\n📊 Plots saved as 'performance_plots.png'")
    # Note: plt.show() might not work in all CLI environments, so we save to file.

    # 8. Interactive Prediction
    print("\n" + "-"*40)
    print("Example Prediction:")
    # Predicting for a house with 1500 sqft, 3 beds, 2 baths
    prediction = model.predict(1500, 3, 2)
    print(f"A 1500 sqft house (3 bed/2 bath) is estimated at: ${prediction:,.2f}")
    print("-"*40)

if __name__ == "__main__":
    train_and_save()
