import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

from training.trainer_base import BaseTrainer, EarlyStopping

class EvaluatorTrainer(BaseTrainer):
    def __init__(self, model, config):
        super().__init__(model, config, criterion=nn.CrossEntropyLoss())
    
    def train(self, train_loader, val_loader):
        """Train the classifier model"""
        optimizer, scheduler = self.setup_optimizer()
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        best_val_loss = float('inf')
        best_state_dict = None
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} - Training")
            for batch in pbar:
                # Get data
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Convert one-hot encoded labels to class indices if needed
                if labels.dim() > 1 and labels.size(1) > 1:
                    targets = torch.argmax(labels, dim=1)
                else:
                    targets = labels
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_val
                    )
                    
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
            train_loss = running_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(self.config.evaluator_best_model_path)
                best_state_dict = self.model.state_dict().copy()
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered")
                break
                
            logging.info(f'Epoch {epoch+1}/{self.config.num_epochs}')
            logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Load best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        
        return self.model
    
    def validate(self, val_loader):
        """Validate the classifier model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Convert one-hot encoded labels to class indices if needed
                if labels.dim() > 1 and labels.size(1) > 1:
                    targets = torch.argmax(labels, dim=1)
                else:
                    targets = labels
                
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, targets)
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                
                # Update statistics
                running_loss += loss.item()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        return val_loss, val_acc
    
    def evaluate(self, val_loader):
        """Evaluate model and return detailed metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Convert one-hot encoded labels to class indices if needed
                if labels.dim() > 1 and labels.size(1) > 1:
                    targets = torch.argmax(labels, dim=1)
                else:
                    targets = labels
                
                outputs = self.model(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist()
        }
        
        return metrics