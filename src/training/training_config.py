import torch
from data.data_loader import DataProcessor
from models.seq_detect import ModernSeqCoreEvaluator, ModernSeqCoreDetector
from utils.inference import InferenceHelper
from utils.eval_metrics import ModelEvaluator
import os
from pathlib import Path
from datetime import datetime

class TrainingConfig:
    def __init__(self):
        # General settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        
        # Data paths
        self.data_dir = Path("data")
        self.processed_data_file = self.data_dir / "processed" / "rossmann_full_seq_processed.pkl"
        self.embeddings_file = self.data_dir / "processed" / "rossmann_full_seq_embeddings.pt"
        
        # Model settings
        self.embedding_dim = 1280  # ESM-2 dimension
        self.detector_embedding_dim = 1280
        self.max_seq_length = 65
        self.dropout = 0.3
        
        # Training settings
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.gradient_clip_val = 1.0
        
        # Early stopping
        self.early_stopping_patience = 10
        
        # Model saving
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"model_outputs/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for ModernSeqCoreEvaluator (classifier)
        self.evaluator_best_model_path = self.output_dir / "evaluator" / "best_evaluator_model.pth"
        self.evaluator_final_model_path = self.output_dir / "evaluator"/ "final_evaluator_model.pth"
        
        # Paths for ModernSeqCoreDetector (sequence tagger)
        self.detector_best_model_path = self.output_dir / "detector" / "best_detector_model.pth"
        self.detector_final_model_path = self.output_dir /"detector"/ "final_detector_model.pth"
        
        # For backward compatibility
        self.best_model_path = self.evaluator_best_model_path
        self.final_model_path = self.evaluator_final_model_path
        
        # Training split 
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        
        # Optimizer settings (for future flexibility)
        self.optimizer_type = "AdamW"  # Could be "Adam", "SGD", etc.
        self.scheduler_type = "ReduceLROnPlateau"  # Could add other schedulers
        self.scheduler_patience = 5
        self.scheduler_factor = 0.5
        
    def save_config(self):
        """Save configuration to a file"""
        import json
        
        # Convert config to dictionary, handling non-serializable types
        config_dict = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v 
                      for k, v in self.__dict__.items()}
        
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
    
    def __str__(self):
        """String representation of config"""
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())