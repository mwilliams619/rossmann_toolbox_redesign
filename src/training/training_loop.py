from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def train_multiclass_model(model, train_loader, val_loader, config):
    """Train a multiclass classification model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), 
                    lr=config.learning_rate, 
                    weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Training")
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
            
            # Gradient clipping
            if hasattr(config, 'gradient_clip_val') and config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        val_loss, val_acc = validate_multiclass_model(model, val_loader, criterion, config)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.evaluator_best_model_path)
            best_state_dict = model.state_dict().copy()
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Load best model before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    return model  # Return the model with best weights

def validate_multiclass_model(model, val_loader, criterion, config):
    """Validate a multiclass classification model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
            _, predicted = torch.max(outputs.data, 1)
            
            # Update statistics
            running_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Calculate average loss and accuracy
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total if total > 0 else 0
    
    return val_loss, val_acc

def evaluate_multiclass_model(model, val_loader, config):
    """Evaluate a multiclass classification model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
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
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist()
    }
    
    return metrics

def train_binary_sequence_tagger(model, train_loader, val_loader, config):
    """Train a binary sequence tagging model"""
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), 
                    lr=config.learning_rate, 
                    weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Training")
        for batch in pbar:
            # Get data
            embeddings = batch['embeddings'].to(config.device)
            labels = batch['sequence_labels'].to(config.device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(embeddings)
            
            # Ensure outputs match labels shape - properly squeeze dimensions
            outputs = outputs.squeeze(-1)  # Remove the last dimension if it's 1
            
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            if hasattr(config, 'gradient_clip_val') and config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        val_loss = validate_binary_sequence_tagger(model, val_loader, criterion, config)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.detector_best_model_path)
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

def validate_binary_sequence_tagger(model, val_loader, criterion, config):
    """Validate a binary sequence tagging model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embeddings'].to(config.device)
            labels = batch['sequence_labels'].to(config.device)
            
            outputs = model(embeddings)
            outputs = outputs.squeeze(-1)  # Remove the last dimension
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
    val_loss = running_loss / len(val_loader)
    return val_loss

def evaluate_binary_sequence_tagger(model, val_loader, config):
    """Evaluate a binary sequence tagging model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embeddings'].to(config.device)
            labels = batch['sequence_labels'].to(config.device)
            
            outputs = model(embeddings)
            outputs = outputs.squeeze(-1)  # Remove the last dimension
            
            # Convert outputs to predictions (threshold at 0.5)
            predictions = (outputs > 0.5).float()
            
            all_preds.append(predictions.cpu())
            all_targets.append(labels.cpu())
    
    # Flatten predictions and targets
    flat_preds = torch.cat([p.view(-1) for p in all_preds])
    flat_targets = torch.cat([t.view(-1) for t in all_targets])
    
    # Calculate metrics
    TP = ((flat_preds == 1) & (flat_targets == 1)).sum().item()
    FP = ((flat_preds == 1) & (flat_targets == 0)).sum().item()
    FN = ((flat_preds == 0) & (flat_targets == 1)).sum().item()
    TN = ((flat_preds == 0) & (flat_targets == 0)).sum().item()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': TP,
        'false_positives': FP,
        'false_negatives': FN,
        'true_negatives': TN
    }
    
    return metrics