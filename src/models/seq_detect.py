import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class SequenceDataset(Dataset):
    def __init__(self, embeddings, labels, dataframe=None):
        self.embeddings = embeddings
        self.labels = labels
        self.dataframe = dataframe

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Get embeddings - handle both numpy arrays and torch tensors
        embedding = self.embeddings[idx]
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        else:
            embedding = embedding.clone().detach().float()
        
        # Get label (always available)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Create return dict
        item = {
            'embeddings': embedding,
            'labels': label
        }
        
        # If dataframe is available, add sequence labels
        if self.dataframe is not None:
            # Get corresponding row from dataframe
            row = self.dataframe.iloc[idx]
            seq = row['seq']
            
            # Process ss_pos to get critical residue positions
            ss_pos = []
            if isinstance(row['ss_pos'], str):
                # Handle string representation of numpy array
                import re
                ss_pos = [int(num) for num in re.findall(r'np\.int64\((\d+)\)', row['ss_pos'])]
            elif isinstance(row['ss_pos'], list):
                # Handle list of positions
                ss_pos = [int(pos) for pos in row['ss_pos']]
            
            # Create binary label tensor
            seq_label = torch.zeros(len(seq))
            for pos in ss_pos:
                if 0 <= pos < len(seq):  # Safety check
                    seq_label[pos] = 1.0
            
            # Add sequence information to return dict
            item['sequence_labels'] = seq_label.float()
            item['seq'] = seq
        
        return item

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

class ModernSeqCoreDetector(nn.Module):
    """Modern version of the SeqCoreDetector for specific position detection"""
    def __init__(self, embedding_dim=640, dropout=0.3):
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
