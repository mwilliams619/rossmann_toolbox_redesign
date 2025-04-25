import torch
from pathlib import Path
import logging

from training.training_config import TrainingConfig
from data.data_loader import DataProcessor
from models.seq_detect import ModernSeqCoreEvaluator
from training.evaluator_trainer import EvaluatorTrainer

def train_evaluator():
    # Initialize configuration
    config = TrainingConfig()
    
    # Setup output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s',
                      handlers=[logging.FileHandler(config.output_dir / "evaluator_training.log"),
                                logging.StreamHandler()])
    
    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Load and prepare data
    data_processor = DataProcessor(config, task_type="evaluation")
    embeddings, labels, dataframe = data_processor.load_data()
    train_loader, val_loader = data_processor.prepare_dataloaders(embeddings, labels, dataframe)
    
    # Initialize model
    model = ModernSeqCoreEvaluator(embedding_dim=config.embedding_dim, dropout=config.dropout)
    
    # Initialize trainer
    trainer = EvaluatorTrainer(model, config)
    
    # Train model
    logging.info("Starting evaluator model training")
    best_model = trainer.train(train_loader, val_loader)
    
    # Save final model
    trainer.save_model(config.evaluator_final_model_path)
    logging.info(f"Final evaluator model saved to {config.evaluator_final_model_path}")
    
    # Evaluate model
    metrics = trainer.evaluate(val_loader)
    logging.info(f"Evaluator model metrics:\n{metrics}")
    
    # Save metrics
    with open(config.output_dir / "evaluator_metrics.txt", "w") as f:
        f.write(str(metrics))
    
    return best_model, metrics

if __name__ == "__main__":
    train_evaluator()