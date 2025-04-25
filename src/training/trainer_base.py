import torch
import torch.nn as nn
from tqdm.auto import tqdm
import logging
from pathlib import Path

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class BaseTrainer:
    def __init__(self, model, config, criterion=None):
        self.model = model
        self.config = config
        self.criterion = criterion
        self.device = config.device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s',
                          handlers=[logging.FileHandler(config.output_dir / "training.log"),
                                    logging.StreamHandler()])
    
    def train(self, train_loader, val_loader):
        """Base training method to be overridden"""
        raise NotImplementedError("Subclasses must implement train method")
    
    def save_model(self, path):
        """Save model to the specified path"""
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            verbose=True
        )
        
        return optimizer, scheduler