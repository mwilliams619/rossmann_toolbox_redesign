from torch.utils.data import Dataset, DataLoader, random_split

class SequenceDataset(Dataset):
    def __init__(self, embeddings, labels):
        """
        Initialize sequence dataset
        
        Args:
            embeddings: torch tensor of shape (num_samples, embedding_dim)
            labels: list or tensor of labels
        """
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        return {
            'embeddings': self.embeddings[idx],
            'labels': self.labels[idx]
        }

class DataProcessor:
    @staticmethod
    def prepare_dataloaders(embeddings, labels, config, split_ratio=0.8):
        """Prepare train and validation dataloaders"""
        dataset = SequenceDataset(embeddings, labels)
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader