from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

def train_epoch(model, train_loader, criterion, optimizer, config):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    
    # Use tqdm for a progress bar
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Get data
        embeddings = batch['embeddings'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        # Convert one-hot encoded labels to class indices if needed
        if labels.dim() > 1 and labels.size(1) > 1:
            targets = torch.argmax(labels, dim=1)
        else:
            targets = labels
            
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(embeddings)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        
        # Optional gradient clipping
        if hasattr(config, 'gradient_clip_val') and config.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    # Calculate average loss over the epoch
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(model, val_loader, criterion, config):
    """Validate the model on validation data"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Get data
            embeddings = batch['embeddings'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Convert one-hot encoded labels to class indices if needed
            if labels.dim() > 1 and labels.size(1) > 1:
                targets = torch.argmax(labels, dim=1)
            else:
                targets = labels
            
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Update statistics
            running_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate average loss
    val_loss = running_loss / len(val_loader)
    return val_loss, all_preds, all_labels

def train_model(model, train_loader, val_loader, config):
    """Complete training pipeline"""
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), 
                    lr=config.learning_rate, 
                    weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config)
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, config)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.best_model_path)
            best_state_dict = model.state_dict().copy()
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Load best model before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    return model  # Return the model with best weights