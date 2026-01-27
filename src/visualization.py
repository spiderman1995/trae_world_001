import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_prediction_analysis(df, save_path="prediction_analysis.png"):
    """
    Generate analysis charts for predictions.
    """
    plt.figure(figsize=(15, 10))
    
    # 1. High Price Prediction vs Target
    plt.subplot(2, 2, 1)
    plt.scatter(df['target_high'], df['pred_high'], alpha=0.5)
    plt.plot([df['target_high'].min(), df['target_high'].max()], 
             [df['target_high'].min(), df['target_high'].max()], 'r--')
    plt.xlabel('Actual High Price')
    plt.ylabel('Predicted High Price (Q90)')
    plt.title('High Price Prediction Accuracy')
    
    # 2. Low Price Prediction vs Target
    plt.subplot(2, 2, 2)
    plt.scatter(df['target_low'], df['pred_low'], alpha=0.5, color='green')
    plt.plot([df['target_low'].min(), df['target_low'].max()], 
             [df['target_low'].min(), df['target_low'].max()], 'r--')
    plt.xlabel('Actual Low Price')
    plt.ylabel('Predicted Low Price (Q10)')
    plt.title('Low Price Prediction Accuracy')
    
    # 3. Direction Confusion Matrix (Simple Bar)
    plt.subplot(2, 2, 3)
    df['pred_dir_class'] = (df['pred_direction_prob'] > 0.5).astype(int)
    confusion = pd.crosstab(df['target_direction'], df['pred_dir_class'])
    confusion.plot(kind='bar', ax=plt.gca())
    plt.title('Direction Prediction Confusion Matrix')
    plt.xlabel('Actual Direction (0: Down, 1: Up)')
    plt.ylabel('Count')
    
    # 4. Sharpe Ratio Distribution
    plt.subplot(2, 2, 4)
    # Filter NaNs
    pred_sharpe = df['pred_sharpe'].dropna()
    target_sharpe = df['target_sharpe'].dropna()
    
    if len(target_sharpe) > 0:
        plt.hist(pred_sharpe, bins=20, alpha=0.5, label='Predicted')
        plt.hist(target_sharpe, bins=20, alpha=0.5, label='Actual')
        plt.legend()
        plt.title('Sharpe Ratio Distribution')
    else:
        plt.text(0.5, 0.5, "No valid Sharpe Ratios (NaN)", ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Analysis plot saved to {save_path}")

if __name__ == "__main__":
    if os.path.exists("predictions.csv"):
        df = pd.read_csv("predictions.csv")
        plot_prediction_analysis(df)
