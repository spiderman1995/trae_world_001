import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models.feature_extractor import FeatureExtractor
from src.models.transformer import StockViT
from src.data.dataset import StockDataset
from torch.utils.data import DataLoader

class Predictor:
    def __init__(self, checkpoint_path, data_dir, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize Models (Use same args as training, or save args in checkpoint)
        # Assuming defaults or hardcoded for now based on training script
        self.embed_dim = 1024 # Matched with training
        self.seq_len = 1 # Matched with test training (was 180)
        self.pred_len = 1 # Matched with test training (was 40)
        
        self.feature_extractor = FeatureExtractor(input_channels=18, output_dim=self.embed_dim).to(self.device)
        self.vit_model = StockViT(seq_len=self.seq_len, embed_dim=self.embed_dim, depth=4, num_heads=4).to(self.device)
        
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.vit_model.load_state_dict(checkpoint['vit'])
        
        self.feature_extractor.eval()
        self.vit_model.eval()
        
    def predict(self, stock_id=None, date=None):
        """
        Run inference for specific stock or all stocks.
        If date is provided, predict for that specific date window.
        """
        # Reuse Dataset for easy data loading
        # In inference mode, we might not have targets, but the dataset class requires them currently.
        # We can use 'val' mode or ignore targets.
        dataset = StockDataset(self.data_dir, seq_len=self.seq_len, pred_len=self.pred_len, mode='val')
        
        # Filter if stock_id is specified
        if stock_id:
            dataset.indices = [idx for idx in dataset.indices if idx[0] == stock_id]
            
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        results = []
        
        with torch.no_grad():
            for seq_data, targets in tqdm(loader, desc="Predicting"):
                # seq_data: [1, 180, 18, 1442]
                B, Seq, C, L = seq_data.shape
                seq_data_flat = seq_data.view(B * Seq, C, L).to(self.device)
                
                # Inference
                features_flat = self.feature_extractor(seq_data_flat)
                features_seq = features_flat.view(B, Seq, -1)
                outputs = self.vit_model(features_seq)
                
                # Parse outputs
                pred_high_ratio = outputs['high'].item()
                pred_low_ratio = outputs['low'].item()
                pred_sharpe = outputs['sharpe'].item()
                pred_dir_prob = torch.sigmoid(outputs['direction']).item()
                
                # Get context info
                # We need to find which stock/date this batch corresponds to
                # Since we iterate sequentially, we can map back if needed, 
                # or modify dataset to return metadata.
                # For simplicity here, we just return the predictions and targets.
                
                current_price = targets['current_price'].item()
                
                results.append({
                    'current_price': current_price,
                    'pred_high': current_price * pred_high_ratio,
                    'pred_low': current_price * pred_low_ratio,
                    'pred_sharpe': pred_sharpe,
                    'pred_direction_prob': pred_dir_prob,
                    'target_high': targets['high'].item() * current_price,
                    'target_low': targets['low'].item() * current_price,
                    'target_sharpe': targets['sharpe'].item(),
                    'target_direction': targets['direction'].item()
                })
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    predictor = Predictor(
        checkpoint_path="runs/model_epoch_1.pth",
        data_dir=r"D:\temp\0_tempdata7"
    )
    df = predictor.predict()
    print(df.head())
    df.to_csv("predictions.csv", index=False)
