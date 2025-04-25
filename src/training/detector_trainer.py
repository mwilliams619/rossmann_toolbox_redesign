import torch
import torch.nn as nn
from tqdm.auto import tqdm
import logging
from sklearn.metrics import precision_score, recall_score, f1_score 
import numpy as np
from training.trainer_base import BaseTrainer, EarlyStopping

class DetectorTrainer(BaseTrainer):
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Loss weights for each task
        self.core_weight = 1.0  
        self.residue_weight = 1.0
        
        # Loss functions (using weighted BCE for imbalanced data)
        self.core_criterion = nn.BCELoss()
        self.residue_criterion = nn.BCELoss()
    
    def train(self, train_loader, val_loader):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        best_val_loss = float('inf')
        best_state_dict = None
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} - Training")
            for batch in pbar:
                embeddings = batch['embeddings'].to(self.device)
                core_labels = batch['core_labels'].to(self.device)
                residue_labels = batch['residue_labels'].to(self.device)
                
                # Forward pass
                core_scores, residue_scores = self.model(embeddings)
                
                # Compute losses
                core_loss = self.core_criterion(core_scores, core_labels)
                residue_loss = self.residue_criterion(residue_scores, residue_labels)
                
                # Combined loss
                loss = self.core_weight * core_loss + self.residue_weight * residue_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
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
            val_loss, val_metrics = self.validate(val_loader)
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = self.model.state_dict().copy()
                self.save_model(self.config.detector_best_model_path)
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
            print(f'Epoch {epoch+1}/{self.config.num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Core Detection F1: {val_metrics["core_f1"]:.4f}, Residue Detection F1: {val_metrics["residue_f1"]:.4f}')

        # Load best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        
        return self.model
    
    def evaluate(self, data_loader):
        """Evaluate the model on the provided data loader and return metrics only."""
        val_loss, val_metrics = self.validate(data_loader)
        return val_metrics
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        
        all_core_preds, all_core_labels = [], []
        all_residue_preds, all_residue_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                core_labels = batch['core_labels'].to(self.device)
                residue_labels = batch['residue_labels'].to(self.device)
                
                core_scores, residue_scores = self.model(embeddings)
                
                # Compute losses
                core_loss = self.core_criterion(core_scores, core_labels)
                residue_loss = self.residue_criterion(residue_scores, residue_labels)
                
                loss = self.core_weight * core_loss + self.residue_weight * residue_loss
                running_loss += loss.item()
                
                # Collect predictions
                all_core_preds.append((core_scores > 0.5).float().cpu())
                all_core_labels.append(core_labels.cpu())
                all_residue_preds.append((residue_scores > 0.5).float().cpu())
                all_residue_labels.append(residue_labels.cpu())
        
        val_loss = running_loss / len(val_loader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            torch.cat(all_core_preds), 
            torch.cat(all_core_labels),
            torch.cat(all_residue_preds), 
            torch.cat(all_residue_labels)
        )
        
        return val_loss, metrics
    
    def _calculate_metrics(self, core_preds, core_labels, residue_preds, residue_labels):
        # Core metrics
        core_precision = precision_score(core_labels.flatten().numpy(), 
                                       core_preds.flatten().numpy(), 
                                       zero_division=0)
        core_recall = recall_score(core_labels.flatten().numpy(), 
                                 core_preds.flatten().numpy(), 
                                 zero_division=0)
        core_f1 = f1_score(core_labels.flatten().numpy(), 
                         core_preds.flatten().numpy(), 
                         zero_division=0)
        
        # Residue metrics
        residue_precision = precision_score(residue_labels.flatten().numpy(), 
                                          residue_preds.flatten().numpy(), 
                                          zero_division=0)
        residue_recall = recall_score(residue_labels.flatten().numpy(), 
                                    residue_preds.flatten().numpy(), 
                                    zero_division=0)
        residue_f1 = f1_score(residue_labels.flatten().numpy(), 
                            residue_preds.flatten().numpy(), 
                            zero_division=0)
        
        return {
            'core_precision': core_precision,
            'core_recall': core_recall,
            'core_f1': core_f1,
            'residue_precision': residue_precision,
            'residue_recall': residue_recall,
            'residue_f1': residue_f1,
        }
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)