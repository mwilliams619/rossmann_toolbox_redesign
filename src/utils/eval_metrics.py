import numpy as np
import torch

class ModelEvaluator:
    def calculate_metrics(self, predictions, labels):
        """
        Calculate evaluation metrics with robust input handling
        
        Args:
            predictions: Model predictions (array or tensor)
            labels: Ground truth labels (array, tensor, or list of batches)
        """
        # Convert predictions to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
            
        # Handle labels that come as a list of batches
        if isinstance(labels, list):
            processed_labels = []
            for batch in labels:
                # Convert tensors to numpy
                if torch.is_tensor(batch):
                    batch = batch.cpu().numpy()
                processed_labels.append(batch)
                
            try:
                # Try to concatenate them
                labels = np.concatenate(processed_labels, axis=0)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not concatenate labels due to {e}.")
                print(f"First batch shape: {np.array(processed_labels[0]).shape}")
                # Fall back to first batch for debugging
                labels = processed_labels[0]
        elif torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        # Ensure shapes match
        if len(predictions) != len(labels):
            raise ValueError(f"Predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")
            
        # Calculate metrics
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
        
        # Add other metrics as needed
        return {
            "accuracy": float(accuracy),
            # Add more metrics here
        }
    
    @staticmethod
    def confusion_matrix(predictions, labels):
        """Generate confusion matrix"""
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        return np.bincount(true_classes * 4 + pred_classes, minlength=16).reshape(4, 4)