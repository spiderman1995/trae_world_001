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
    def __init__(self, checkpoint_path, data_dir, device=None, seq_len=None, pred_len=None, embed_dim=None, depth=None, num_heads=None, input_channels=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        cfg = checkpoint.get("config") or {}
        self.embed_dim = embed_dim or cfg.get("embed_dim") or 1024
        self.seq_len = seq_len or cfg.get("seq_len") or 180
        self.pred_len = pred_len or cfg.get("pred_len") or 60
        self.depth = depth or cfg.get("depth") or 4
        self.num_heads = num_heads or cfg.get("num_heads") or 4
        self.input_channels = input_channels or cfg.get("input_channels") or 18
        self.mean = checkpoint.get("mean")
        self.std = checkpoint.get("std")

        self.feature_extractor = FeatureExtractor(input_channels=self.input_channels, output_dim=self.embed_dim).to(self.device)
        self.vit_model = StockViT(seq_len=self.seq_len, pred_len=self.pred_len, embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads).to(self.device)
        
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
        dataset = StockDataset(self.data_dir, seq_len=self.seq_len, pred_len=self.pred_len,
                              mean=self.mean, std=self.std)
        
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
                pred_max_value_ratio = outputs['max_value'].item()
                pred_min_value_ratio = outputs['min_value'].item()
                pred_max_day = torch.softmax(outputs['max_day'], dim=1).argmax(dim=1).item()
                pred_min_day = torch.softmax(outputs['min_day'], dim=1).argmax(dim=1).item()
                
                # Get context info
                # We need to find which stock/date this batch corresponds to
                # Since we iterate sequentially, we can map back if needed, 
                # or modify dataset to return metadata.
                # For simplicity here, we just return the predictions and targets.
                
                current_price = targets['current_price'].item()
                
                results.append({
                    'current_price': current_price,
                    'pred_max_value_ratio': pred_max_value_ratio,
                    'pred_min_value_ratio': pred_min_value_ratio,
                    'pred_max_value': current_price * pred_max_value_ratio,
                    'pred_min_value': current_price * pred_min_value_ratio,
                    'pred_max_day': pred_max_day,
                    'pred_min_day': pred_min_day,
                    'target_max_value_ratio': targets['max_value'].item(),
                    'target_min_value_ratio': targets['min_value'].item(),
                    'target_max_value': targets['max_value'].item() * current_price,
                    'target_min_value': targets['min_value'].item() * current_price,
                    'target_max_day': targets['max_day'].item(),
                    'target_min_day': targets['min_day'].item()
                })
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    from src.config import DATA_DIR
    predictor = Predictor(
        checkpoint_path="runs/model_epoch_1.pth",
        data_dir=DATA_DIR
    )
    df = predictor.predict()
    print(df.head())
    df.to_csv("predictions.csv", index=False)
