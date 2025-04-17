import torch
import pickle
from pathlib import Path
import logging
from datetime import datetime
import torch.nn as nn

from training.training_config import TrainingConfig
from utils.data_processing import DataProcessor
from src.models.seq_detect import ModernSeqCoreEvaluator, ModernSeqCoreDetector
from training.training_loop import (
    train_multiclass_model,
    train_binary_sequence_tagger,
    evaluate_multiclass_model,
    evaluate_binary_sequence_tagger
)
from utils.inference import InferenceHelper
from utils.eval_metrics import ModelEvaluator

def setup_logging(config):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(config.output_dir / "training.log"),
                                  logging.StreamHandler()])

def main():
    try:
        import pandas as pd
        import numpy as np
        import re
        import io
        
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
        
        # Debug what's in the data
        logging.info(f"Data type: {type(data)}")

        logging.info(f"Data dictionary keys: {list(data.keys())}")
        dataframe = None  # Initialize as None

        # Option 1: Look for DataFrame directly in dictionary values
        for key, value in data.items():
            if key != 'labels' and isinstance(value, pd.DataFrame):
                logging.info(f"Found DataFrame under key '{key}' with columns: {list(value.columns)}")
                if 'seq' in value.columns and 'ss_pos' in value.columns:
                    dataframe = value
                    logging.info(f"Using DataFrame from key '{key}' with shape {dataframe.shape}")
                    break

        # Option 2: Look for a key that might contain the sequence data
        if dataframe is None and 'sequences' in data:
            seq_data = data['sequences']
            logging.info(f"Found 'sequences' key with type: {type(seq_data)}")
            if isinstance(seq_data, pd.DataFrame):
                dataframe = seq_data
                logging.info(f"Using DataFrame from 'sequences' key with shape {dataframe.shape}")
            elif isinstance(seq_data, list) and len(seq_data) > 0:
                # Try to convert list to DataFrame if it has the right structure
                try:
                    dataframe = pd.DataFrame(seq_data)
                    logging.info(f"Converted list from 'sequences' key to DataFrame with shape {dataframe.shape}")
                except:
                    logging.error("Failed to convert list from 'sequences' key to DataFrame")

        # If still None, look for any other potential keys
        if dataframe is None:
            for key in data.keys():
                if key != 'labels' and isinstance(data[key], (list, dict)):
                    logging.info(f"Found potential data structure in key '{key}' with type {type(data[key])}")
        
        # If data is a string that looks like CSV, parse it as a DataFrame
        if isinstance(data, str) and ',' in data:
            logging.info("Data appears to be CSV text. Attempting to parse as DataFrame...")
            try:
                # Try to parse as CSV
                dataframe = pd.read_csv(io.StringIO(data))
                logging.info(f"Successfully parsed CSV data into DataFrame with columns: {list(dataframe.columns)}")
                
                # For this case, we need to create class labels
                # You may need to adjust this based on your actual data structure
                if 'simplified_cofactor' in dataframe.columns:
                    class_labels = dataframe['simplified_cofactor'].tolist()
                    logging.info(f"Using 'simplified_cofactor' column as labels, found {len(set(class_labels))} unique classes")
                else:
                    logging.error("No suitable column for class labels")
                    return
                
            except Exception as e:
                logging.error(f"Failed to parse CSV data: {e}")
                return
                
        elif isinstance(data, dict):
            # Extract class labels
            if 'labels' in data:
                class_labels = data['labels']
                logging.info(f"Found labels in data dict, length: {len(class_labels)}")
            else:
                logging.error("No 'labels' key found in data dict")
                return
            
            # Try to find the DataFrame
            dataframe = None
            
            # First, check if any value is a DataFrame
            for key, value in data.items():
                if key != 'labels' and isinstance(value, pd.DataFrame):
                    if 'seq' in value.columns and 'ss_pos' in value.columns:
                        dataframe = value
                        logging.info(f"Found DataFrame in key '{key}' with required columns")
                        break
            
            # If no DataFrame found, check if any value is a string that could be CSV
            if dataframe is None:
                for key, value in data.items():
                    if key != 'labels' and isinstance(value, str) and ',' in value:
                        try:
                            df = pd.read_csv(io.StringIO(value))
                            if 'seq' in df.columns and 'ss_pos' in df.columns:
                                dataframe = df
                                logging.info(f"Parsed CSV data from key '{key}' into DataFrame")
                                break
                        except:
                            pass
        
        # As a last resort, if your pickle contains the CSV content directly
        if dataframe is None and isinstance(data, str) and 'pdb_chain,seq,seqres_start' in data:
            try:
                # This looks like the CSV header from your example
                logging.info("Attempting to parse direct CSV content...")
                dataframe = pd.read_csv(io.StringIO(data))
                
                # We need to extract labels somehow - you'll need to adjust this
                # based on your specific data structure
                if 'simplified_cofactor' in dataframe.columns:
                    class_labels = dataframe['simplified_cofactor'].tolist()
                    logging.info(f"Using 'simplified_cofactor' column as labels")
                else:
                    # Try to infer labels from other columns if needed
                    logging.warning("No label column found, might need manual extraction")
            except Exception as e:
                logging.error(f"Failed to parse direct CSV content: {e}")
        
        # Load embeddings
        embeddings = torch.load(embeddings_path)
        logging.info(f"Loaded embeddings of shape/length: {len(embeddings)}")
        
        # Process ss_pos column if it contains string representations
        if dataframe is not None and 'ss_pos' in dataframe.columns:
            logging.info("Processing ss_pos column")
            
            # Create a function to convert string representation to list of integers
            def convert_ss_pos(pos_str):
                if isinstance(pos_str, str):
                    try:
                        # Extract numbers from the string using regex
                        numbers = re.findall(r'np\.int64\((\d+)\)', pos_str)
                        return [int(num) for num in numbers]
                    except:
                        return []
                return pos_str  # Return unchanged if not a string
            
            # Apply the conversion
            dataframe['ss_pos'] = dataframe['ss_pos'].apply(convert_ss_pos)
            logging.info(f"Processed ss_pos sample: {dataframe['ss_pos'].iloc[0]}")
            
            # Always store a copy of the processed dataframe for debugging
            dataframe.to_csv(config.output_dir / "processed_dataframe.csv", index=False)
            logging.info(f"Saved processed dataframe to {config.output_dir / 'processed_dataframe.csv'}")
        
        # Create data processor and prepare dataloaders
        data_processor = DataProcessor()
        
        if dataframe is not None:
            logging.info("Using unified dataloaders for both models")
            # Create unified dataloaders that provide both class labels and sequence labels
            train_loader, val_loader = data_processor.prepare_dataloaders(
                embeddings, class_labels, dataframe, config
            )
            have_sequence_data = True
        else:
            logging.info("Using classifier-only dataloaders")
            # Fall back to classifier-only dataloaders
            train_loader, val_loader = data_processor.prepare_dataloaders(
                embeddings, class_labels, None, config
            )
            have_sequence_data = False
        
        for batch in train_loader:
            logging.info(f"Batch keys: {batch.keys()}")
            logging.info(f"Sample sequence: {batch['seq'][0]}")
            logging.info(f"Sample sequence labels (ss_pos): {batch['ss_pos'][0]}")
            break
        
        # Save initial configuration
        config.save_config()
        
        # PART 1: Train the classifier (ModernSeqCoreEvaluator)
        logging.info("Training ModernSeqCoreEvaluator (classifier)")
        
        # Initialize classifier model
        logging.info("Initializing classifier model")
        evaluator_model = ModernSeqCoreEvaluator(embedding_dim=config.embedding_dim).to(config.device)
        
        # Train classifier model
        logging.info("Starting classifier model training")
        best_evaluator = train_multiclass_model(evaluator_model, train_loader, val_loader, config)
        
        # Save the final classifier model
        torch.save(best_evaluator.state_dict(), config.evaluator_final_model_path)
        logging.info(f"Final classifier model saved to {config.evaluator_final_model_path}")
        
        # Make predictions with classifier
        logging.info("Making predictions on validation set with classifier")
        evaluator_val_metrics = evaluate_multiclass_model(best_evaluator, val_loader, config)
        logging.info(f"Classifier evaluation metrics:\n{evaluator_val_metrics}")
        
        # Save classifier metrics
        with open(config.output_dir / "classifier_metrics.txt", "w") as f:
            f.write(str(evaluator_val_metrics))
            
        # PART 2: Train the sequence tagger (ModernSeqCoreDetector)
        if have_sequence_data:
            logging.info("Training ModernSeqCoreDetector (critical residue detector)")
            
            # Initialize sequence tagger model
            logging.info("Initializing critical residue detector model")
            detector_model = ModernSeqCoreDetector(embedding_dim=config.embedding_dim).to(config.device)
            
            # Train sequence tagger model using same dataloaders
            logging.info("Starting critical residue detector training")
            best_detector = train_binary_sequence_tagger(detector_model, train_loader, val_loader, config)
            
            # Save the final sequence tagger model
            torch.save(best_detector.state_dict(), config.detector_final_model_path)
            logging.info(f"Final critical residue detector model saved to {config.detector_final_model_path}")
            
            # Evaluate sequence tagger
            logging.info("Evaluating critical residue detector performance")
            detector_metrics = evaluate_binary_sequence_tagger(best_detector, val_loader, config)
            logging.info(f"Critical residue detector metrics:\n{detector_metrics}")
            
            # Save metrics
            with open(config.output_dir / "critical_residue_detector_metrics.txt", "w") as f:
                f.write(str(detector_metrics))
            
            logging.info("Critical residue detector evaluation completed")
        else:
            logging.warning("No dataframe available. Skipped critical residue detector training.")
        
        logging.info("Training process completed successfully")
        
    except Exception as e:
        logging.exception(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()