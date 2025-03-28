import torch
import numpy as np
from pathlib import Path
from src.models.seq_detect import ModernSeqCoreEvaluator
from src.utils.embeddings import ProteinEmbedding

class SequencePredictor:
    def __init__(self, model_path, embedding_dim=640, device=None):
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ModernSeqCoreEvaluator(embedding_dim=embedding_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize embedding generator
        self.embedding_generator = ProteinEmbedding(device=self.device)
    
    def predict(self, sequence):
        """Predict class for a single sequence"""
        return self.predict_batch([sequence])[0]
    
    def predict_batch(self, sequences):
        """Predict classes for a batch of sequences"""
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(sequences)
        
        # Convert to tensor
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float)
        embeddings = embeddings.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(embeddings)
        
        # Process results
        results = []
        for pred in predictions.cpu().numpy():
            class_id = np.argmax(pred)
            results.append({
                'class_id': class_id,
                'class_name': self.get_class_name(class_id),
                'confidence': float(pred[class_id]),
                'probabilities': pred.tolist()
            })
        
        return results
    
    def get_class_name(self, class_id):
        """Map numeric class ID to human-readable name"""
        class_names = ["FAD", "NAD", "NADP", "SAM"]  # Replace with your actual class names
        return class_names[class_id]

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    model_path = "model_outputs/20250325_213422/best_model.pth"  # Update with your model path
    predictor = SequencePredictor(model_path)
    
    # Example sequences
    test_sequences = [
        'AGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITDIDEGR', # WT, binds NAD
        'AGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITSRDEGR' # D211S, I212R mutant, binds NADP
    ]
    
    # Get predictions
    for i, sequence in enumerate(test_sequences):
        result = predictor.predict(sequence)
        print(f"Sequence {i+1}:")
        print(f"  Predicted: {result['class_name']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  All probabilities: {[f'{p:.4f}' for p in result['probabilities']]}")
        print()