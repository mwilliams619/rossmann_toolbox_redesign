import torch
from pathlib import Path
import logging

from training.training_config import TrainingConfig
from data.data_loader import DataProcessor
from models.seq_detect import ModernSeqCoreDetector
from training.detector_trainer import DetectorTrainer

def train_detector():
    # Initialize configuration
    config = TrainingConfig()
    
    # Setup output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s',
                      handlers=[logging.FileHandler(config.output_dir / "detector_training.log"),
                                logging.StreamHandler()])
    
    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Load and prepare data
    data_processor = DataProcessor(config, task_type="detection")
    embeddings, processed_data = data_processor.load_data()

    
    train_loader, val_loader = data_processor.prepare_dataloaders(embeddings, processed_data)
    
    # Initialize model
    model = ModernSeqCoreDetector(embedding_dim=config.detector_embedding_dim)
    
    # Initialize trainer
    trainer = DetectorTrainer(model, config)
    
    # Train model
    logging.info("Starting detector model training")
    best_model = trainer.train(train_loader, val_loader)
    
    # Save final model
    trainer.save_model(config.detector_final_model_path)
    logging.info(f"Final detector model saved to {config.detector_final_model_path}")
    
    # Evaluate model
    metrics = trainer.evaluate(val_loader)
    logging.info(f"Detector model metrics:\n{metrics}")
    
    # Save metrics
    with open(config.output_dir / "detector_metrics.txt", "w") as f:
        f.write(str(metrics))
    
    return best_model, metrics

if __name__ == "__main__":
    train_detector()