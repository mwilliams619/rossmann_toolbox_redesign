import torch
import pickle
from pathlib import Path
import logging
from datetime import datetime

from training.training_config import TrainingConfig
from utils.data_processing import DataProcessor
from src.models.seq_detect import ModernSeqCoreEvaluator
from training.training_loop import train_model
from utils.inference import InferenceHelper
from utils.eval_metrics import ModelEvaluator

def setup_logging(config):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(config.output_dir / "training.log"),
                                  logging.StreamHandler()])

def main():
    try:
        # Initialize configuration
        config = TrainingConfig()
        
        # Setup logging
        setup_logging(config)
        
        logging.info("Starting training process")
        logging.info(f"Configuration:\n{config}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Load data
        logging.info("Loading data")
        data_path = Path(config.processed_data_file)
        embeddings_path = Path(config.embeddings_file)
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        labels = data['labels']
        embeddings = torch.load(embeddings_path)
        
        # Prepare data
        logging.info("Preparing data loaders")
        data_processor = DataProcessor()
        train_loader, val_loader = data_processor.prepare_dataloaders(
            embeddings, labels, config
        )
        
        # Initialize model
        logging.info("Initializing model")
        model = ModernSeqCoreEvaluator(embedding_dim=config.embedding_dim).to(config.device)
        
        # Save initial configuration
        config.save_config()
        
        # Train model
        logging.info("Starting model training")
        best_model = train_model(model, train_loader, val_loader, config)
        
        # Save the final model
        torch.save(model.state_dict(), config.final_model_path)
        logging.info(f"Final model saved to {config.final_model_path}")
        
        # Make predictions
        logging.info("Making predictions on validation set")
        inference_helper = InferenceHelper(best_model, config)
        predictions = inference_helper.predict_batch(val_loader)
        
        # Evaluate results
        logging.info("Evaluating model performance")
        evaluator = ModelEvaluator()
        val_labels = [batch['labels'] for batch in val_loader]
        metrics = evaluator.calculate_metrics(predictions, val_labels)
        logging.info(f"Evaluation metrics:\n{metrics}")
        
        # Save metrics
        with open(config.output_dir / "metrics.txt", "w") as f:
            f.write(str(metrics))
        
        logging.info("Training process completed successfully")
        
    except Exception as e:
        logging.exception(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()