import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

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


class DataProcessor:
    def prepare_dataloaders(self, embeddings, labels, dataframe=None, config=None):
        """Prepare dataloaders for training and validation"""
        from torch.utils.data import DataLoader, random_split
        import numpy as np
        
        # Create dataset
        if dataframe is not None:
            dataset = SequenceDataset(embeddings, labels, dataframe)
            have_sequence_data = True
        else:
            dataset = SequenceDataset(embeddings, labels)
            have_sequence_data = False
        
        # Split dataset
        train_size = int(len(dataset) * config.train_ratio)
        val_size = len(dataset) - train_size
        
        # Use fixed seed for reproducible splits
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader