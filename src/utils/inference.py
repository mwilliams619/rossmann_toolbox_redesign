import torch
import numpy as np

class InferenceHelper:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model.eval()
        
    @torch.no_grad()
    def predict(self, embeddings):
        """Make predictions for given embeddings"""
        embeddings = torch.tensor(embeddings).float().to(self.config.device)
        outputs = self.model(embeddings)
        return outputs.cpu().numpy()
    
    def predict_batch(self, dataloader):
        """Make predictions for entire dataset"""
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embeddings'].to(self.config.device)
                outputs = self.model(embeddings)
                all_predictions.extend(outputs.cpu().numpy())
                
        return np.array(all_predictions)