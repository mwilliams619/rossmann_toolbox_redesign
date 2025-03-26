import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class SequenceDataset(Dataset):
    """Dataset for protein sequence embeddings"""
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Check if embeddings is already a tensor
        if isinstance(self.embeddings[idx], torch.Tensor):
            embeddings = self.embeddings[idx].clone().detach().float()
        else:
            embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float)
        
        # Ensure we have a sequence dimension - THIS IS THE KEY FIX
        if len(embeddings.shape) == 1:  # If just (embedding_dim,)
            embeddings = embeddings.unsqueeze(0)  # Make it (1, embedding_dim)

        item = {'embeddings': embeddings}
        
        if self.labels is not None:
            if isinstance(self.labels[idx], torch.Tensor):
                item['labels'] = self.labels[idx].clone().detach().float()
            else:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                
        return item

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for enhancing informative features"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
        nn.Linear(channels, channels // reduction, bias=False),
        nn.GELU(),
        nn.Linear(channels // reduction, channels, bias=False),
        nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualConvBlock(nn.Module):
    """Residual block with 1D convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualConvBlock, self).__init__()
        self.same_channels = in_channels == out_channels
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                            padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection if channel dimensions change
        if not self.same_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        
        self.act2 = nn.GELU()
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Apply shortcut if needed
        if not self.same_channels:
            residual = self.shortcut(residual)
            
        x += residual
        x = self.act2(x)
        
        return x

class SelfAttention1D(nn.Module):
    """Self-attention module for 1D sequence data"""
    def __init__(self, channels, reduction=8):
        super(SelfAttention1D, self).__init__()
        self.query = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width = x.size()
        
        # Reshape query, key, value
        proj_query = self.query(x).permute(0, 2, 1)  # B x W x C'
        proj_key = self.key(x)  # B x C' x W
        proj_value = self.value(x)  # B x C x W
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)  # B x W x W
        attention = self.softmax(energy)  # B x W x W
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x W
        
        # Add weighted attention to input
        out = self.gamma * out + x
        
        return out

class ModernSeqCoreEvaluator(nn.Module):
    """Modern version of the SeqCoreEvaluator using advanced components"""
    def __init__(self, embedding_dim=640, dropout=0.3):
        super(ModernSeqCoreEvaluator, self).__init__()

        # Input dimension adjustment (from ESM-2's 768 to processing dimension)
        self.input_proj = nn.Conv1d(embedding_dim, 128, kernel_size=1)
        
        # First block - residual convolutions
        self.block1 = nn.Sequential(
            ResidualConvBlock(128, 64, kernel_size=7),
            SEBlock(64),
            nn.Dropout(dropout)
        )
        
        # Second block - residual convolutions
        self.block2 = nn.Sequential(
            ResidualConvBlock(64, 64, kernel_size=5),
            SEBlock(64),
            nn.Dropout(dropout)
        )
        
        # Attention module for capturing long-range dependencies
        self.attention = SelfAttention1D(64)
        
        # Pooling layer and MLP for classification
        self.norm = nn.LayerNorm(64)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 4)
        )
    
    def forward(self, x):
        # Input shape: (batch_size, seq_length, embedding_dim)
        
        # Check if we're missing the sequence dimension
        if len(x.shape) == 2:  # If shape is (batch_size, embedding_dim) 
        # Add a sequence dimension of length 1
            x = x.unsqueeze(1)  # Now: (batch_size, 1, embedding_dim)
        
        # Convert to channels-first for convolutions
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_length)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Apply convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling and classification
        x = self.pool(x).squeeze(-1)
        x = self.norm(x)
        x = self.classifier(x)
        
        return F.softmax(x, dim=1)

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network from Transformer architecture"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.norm(x + residual)

class ModernSeqCoreDetector(nn.Module):
    """Modern version of the SeqCoreDetector for specific position detection"""
    def __init__(self, embedding_dim=768, dropout=0.3):
        super(ModernSeqCoreDetector, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(embedding_dim, 128)
        self.input_norm = nn.LayerNorm(128)
        
        # Convolutional blocks (channels-first format)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            SEBlock(64),
            
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            SEBlock(64)
        )
    
        # Attention mechanism
        self.attention = SelfAttention1D(64)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input shape: (batch_size, seq_length, embedding_dim)
        
        # Initial projection (batch_size, seq_length, 128)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # Convert to channels-first for convolutions
        x = x.transpose(1, 2)  # (batch_size, 128, seq_length)
        
        # Apply convolutional blocks
        x = self.conv_blocks(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Convert back to sequence-first for per-position classification
        x = x.transpose(1, 2)  # (batch_size, seq_length, 64)
        
        # Apply classifier to each position
        x = self.classifier(x)  # (batch_size, seq_length, 1)
        
        return x

class ModernPreprocessedEmbeddingModel(nn.Module):
    """
    Model that accepts pre-computed embeddings directly
    (to work with the user's preprocessing pipeline)
    """
    def __init__(self, embedding_dim=768, dropout=0.3):
        super(ModernPreprocessedEmbeddingModel, self).__init__()

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(256, 256),
            ResidualMLP(256, 128),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        # Input is pre-computed embeddings
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return F.softmax(logits, dim=1)

class ResidualMLP(nn.Module):
    """Residual MLP block"""
    def __init__(self, in_features, out_features, dropout=0.1):
        super(ResidualMLP, self).__init__()
        self.same_dim = in_features == out_features

        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout2 = nn.Dropout(dropout)
        
        if not self.same_dim:
            self.shortcut = nn.Linear(in_features, out_features)
            
    def forward(self, x):
        residual = x
        
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        
        if not self.same_dim:
            residual = self.shortcut(residual)
            
        x = x + residual
        return F.gelu(x)

class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple predictions
    """
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            # Equal weighting
            weights = [1.0/len(models)] * len(models)
            
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float), 
                                requires_grad=True)
        
    def forward(self, x):
        # Get predictions from all models
        preds = [model(x) for model in self.models]
        preds = torch.stack(preds, dim=0)  # (num_models, batch_size, num_classes)
        
        # Normalize weights
        weights = F.softmax(self.weights, dim=0)
        
        # Compute weighted average
        weighted_preds = weights.view(-1, 1, 1) * preds
        ensemble_pred = weighted_preds.sum(dim=0)
        
        return ensemble_pred