import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class CoreDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, processed_data, max_length=None):
        self.embeddings = embeddings
        self.core_labels = processed_data['core_labels']
        self.residue_labels = processed_data['residue_labels']
        self.dataframe = processed_data['dataframe']
        self.max_length = max_length or self.core_labels.shape[1]
        
        # Validate embeddings shape
        sample = self.embeddings[0]
        expected_shape = (self.max_length, -1)  # [seq_len, embedding_dim]
        
        if len(sample.shape) != 2:
            raise ValueError(f"Expected per-residue embeddings with shape [seq_len, embedding_dim], "
                           f"but got shape {sample.shape}")
        
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        
        # Ensure consistent sequence length
        if embedding.shape[0] > self.max_length:
            embedding = embedding[:self.max_length]
        elif embedding.shape[0] < self.max_length:
            # Pad with zeros
            pad_size = self.max_length - embedding.shape[0]
            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding)
            embedding = torch.nn.functional.pad(embedding, (0, 0, 0, pad_size))
        
        item = {
            'embeddings': torch.tensor(embedding, dtype=torch.float),
            'core_labels': torch.tensor(self.core_labels[idx], dtype=torch.float),
            'residue_labels': torch.tensor(self.residue_labels[idx], dtype=torch.float),
            'pdb_chain': self.dataframe.iloc[idx]['pdb_chain'],
            'core_start': self.dataframe.iloc[idx]['core_start'],
            'core_end': self.dataframe.iloc[idx]['core_end']
        }
        return item
    
    def __len__(self):
        return len(self.dataframe)

class SequenceDataset(Dataset):
    def __init__(self, embeddings, labels=None, dataframe=None):
        # Modified to be more robust and safer
        
        # Handle different embeddings formats consistently
        if isinstance(embeddings, dict):
            self.pdb_chains = list(embeddings.keys())
            self.embeddings_list = [embeddings[key] for key in self.pdb_chains]
        else:
            self.embeddings_list = embeddings
            
            # Ensure we have consistent pdb_chains
            if dataframe is not None:
                if 'pdb_chain' in dataframe.columns:
                    self.pdb_chains = dataframe['pdb_chain'].tolist()
                else:
                    self.pdb_chains = [f"item_{i}" for i in range(len(embeddings))]
            else:
                self.pdb_chains = [f"item_{i}" for i in range(len(embeddings))]
        
        # Critical length validation
        assert len(self.embeddings_list) > 0, "Empty embeddings list"
        
        if labels is not None:
            assert len(self.embeddings_list) == len(labels), f"Length mismatch: {len(self.embeddings_list)} embeddings vs {len(labels)} labels"
            
        if dataframe is not None:
            assert len(self.embeddings_list) == len(dataframe), f"Length mismatch: {len(self.embeddings_list)} embeddings vs {len(dataframe)} rows"
        
        # Validate embeddings format for evaluation (should be global)
        sample = self.embeddings_list[0]
        if isinstance(sample, np.ndarray):
            sample = torch.tensor(sample)
            
        if len(sample.shape) != 1:
            print(f"WARNING: SequenceDataset expected 1D embeddings but got shape {sample.shape}")
            
        self.labels = labels
        self.dataframe = dataframe
        
        print(f"SequenceDataset initialized with {len(self)} embeddings")
    
    def __len__(self):
        return len(self.embeddings_list)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self)} samples")
            
        embedding = self.embeddings_list[idx]
        
        # Ensure tensor format consistently
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float)
        else:
            embedding = embedding.float()
        
        # Create return dictionary
        item = {
            'embeddings': embedding,
            'idx': idx
        }
        
        # Add PDB chain if available
        if hasattr(self, 'pdb_chains') and idx < len(self.pdb_chains):
            item['pdb_chain'] = self.pdb_chains[idx]
        
        # Add label if available
        if self.labels is not None:
            if idx >= len(self.labels):
                raise IndexError(f"Label index {idx} exceeds labels length {len(self.labels)}")
                
            label = self.labels[idx]
            if isinstance(label, np.ndarray):
                item['labels'] = torch.tensor(label, dtype=torch.float)
            elif isinstance(label, torch.Tensor):
                item['labels'] = label.float()
            else:
                item['labels'] = torch.tensor(label, dtype=torch.long)
        
        return item