import torch
import pickle
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd
import numpy as np

from data.dataset import SequenceDataset, CoreDetectionDataset

class DataProcessor:
    def __init__(self, config, task_type="evaluation"):
        self.config = config
        self.task_type = task_type
        print(f"Initialized DataProcessor for task: {task_type}")
    
    def _normalize_embeddings_format(self, embeddings):
        """Standardize embeddings to list format with proper metadata"""
        if isinstance(embeddings, dict):
            # Dictionary format: {pdb_chain: embedding_tensor}
            pdb_chains = list(embeddings.keys())
            embeddings_list = list(embeddings.values())
        else:
            # List format: [embedding_tensor_1, embedding_tensor_2, ...]
            embeddings_list = embeddings
            pdb_chains = None
        
        return embeddings_list, pdb_chains
    
    def _validate_embeddings_shape(self, embeddings_list, expected_dims=None):
        """Check embeddings shape for task compatibility"""
        if not embeddings_list:
            raise ValueError("Empty embeddings list")
            
        sample = embeddings_list[0]
        if isinstance(sample, np.ndarray):
            sample = torch.tensor(sample)
            
        shape_msg = f"Sample embedding shape: {sample.shape}"
        print(shape_msg)
        
        if expected_dims and len(sample.shape) != expected_dims:
            if self.task_type == "detection" and len(sample.shape) == 1:
                print("WARNING: Detection task requires per-residue embeddings [seq_len, embedding_dim]")
                print("         Received global embeddings [embedding_dim]")
                return False
            elif self.task_type == "evaluation" and len(sample.shape) == 2:
                print("WARNING: Evaluation task requires global embeddings [embedding_dim]")
                print("         Received per-residue embeddings [seq_len, embedding_dim]")
                return False
        
        return True
            
    def _convert_embeddings_for_task(self, embeddings_list, processed_data=None):
        """Convert embeddings to the correct format for the task"""
        sample = embeddings_list[0]
        if isinstance(sample, np.ndarray):
            sample = torch.tensor(sample)
        
        # DETECTION: Needs [seq_len, embedding_dim] format
        if self.task_type == "detection" and len(sample.shape) == 1:
            print("Converting global embeddings to per-residue for detection")
            if processed_data is None:
                raise ValueError("Need processed_data to determine sequence length for conversion")
                
            max_length = processed_data['core_labels'].shape[1]
            converted_embeddings = []
            
            for emb in embeddings_list:
                if isinstance(emb, np.ndarray):
                    emb = torch.tensor(emb)
                # Replicate global embedding across sequence length
                per_residue = emb.unsqueeze(0).expand(max_length, -1)
                converted_embeddings.append(per_residue)
                
            print(f"Converted {len(embeddings_list)} global embeddings to per-residue format")
            print(f"New shape: {converted_embeddings[0].shape}")
            return converted_embeddings
        
        # EVALUATION: Needs [embedding_dim] format
        elif self.task_type == "evaluation" and len(sample.shape) == 2:
            print("Converting per-residue embeddings to global for evaluation")
            converted_embeddings = []
            
            for emb in embeddings_list:
                if isinstance(emb, np.ndarray):
                    emb = torch.tensor(emb)
                # Mean pool across sequence dimension
                global_emb = emb.mean(dim=0)
                converted_embeddings.append(global_emb)
                
            print(f"Converted {len(embeddings_list)} per-residue embeddings to global format")
            print(f"New shape: {converted_embeddings[0].shape}")
            return converted_embeddings
        
        # Already in correct format
        return embeddings_list
    
    def load_data(self):
        """Load data with proper validation"""
        data_path = Path(self.config.processed_data_file)
        embeddings_path = Path(self.config.embeddings_file)
        
        # Load processed data
        with open(data_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        # Load embeddings
        embeddings_data = torch.load(embeddings_path)
        
        # Handle metadata if present
        if isinstance(embeddings_data, dict) and 'embeddings' in embeddings_data:
            embeddings = embeddings_data['embeddings']
            metadata = embeddings_data.get('metadata', {})
            print(f"Loaded embeddings format: {metadata.get('format', 'unknown')}")
        else:
            embeddings = embeddings_data
        
        # For detection task
        if self.task_type == "detection":
            # Handle dictionary format
            if isinstance(embeddings, dict):
                # Get chains from the processed data
                chains_in_df = processed_data['dataframe']['pdb_chain'].tolist()
                matching_embeddings = []
                
                for chain in chains_in_df:
                    if chain in embeddings:
                        emb = embeddings[chain]
                        # Ensure we have per-residue embeddings for detection
                        if emb.dim() == 1:  # global embedding
                            # Expand to per-residue format
                            max_length = processed_data['core_labels'].shape[1]
                            emb = emb.unsqueeze(0).expand(max_length, -1)
                        matching_embeddings.append(emb)
                    else:
                        raise ValueError(f"Chain {chain} not found in embeddings")
                
                embeddings_list = matching_embeddings
            
            # Handle tensor format
            elif isinstance(embeddings, torch.Tensor):
                if embeddings.dim() == 3:  # [batch, seq_len, dim]
                    embeddings_list = [embeddings[i] for i in range(embeddings.size(0))]
                elif embeddings.dim() == 2:  # [batch, dim] - global embeddings
                    # Convert to per-residue for detection
                    max_length = processed_data['core_labels'].shape[1]
                    embeddings_list = [emb.unsqueeze(0).expand(max_length, -1) for emb in embeddings]
                else:
                    raise ValueError(f"Unexpected tensor shape: {embeddings.shape}")
            else:
                # Assume list format
                embeddings_list = embeddings
            
            # Validate alignment for detection
            expected_count = len(processed_data['dataframe'])
            if len(embeddings_list) != expected_count:
                raise ValueError(f"Mismatch: {len(embeddings_list)} embeddings vs {expected_count} dataframe rows")
            
            return embeddings_list, processed_data
        
        # For evaluation task
        else:
            labels = processed_data['labels']
            dataframe = processed_data['dataframe']
            
            # Handle dictionary format
            if isinstance(embeddings, dict):
                # Make sure we're using the right PDB chains
                chains_in_df = dataframe['pdb_chain'].tolist()
                matching_embeddings = []
                missing_chains = []
                
                for chain in chains_in_df:
                    if chain in embeddings:
                        emb = embeddings[chain]
                        # Convert to global if needed
                        if emb.dim() == 2:  # per-residue
                            emb = emb.mean(dim=0)
                        elif emb.dim() != 1:
                            raise ValueError(f"Unexpected embedding shape for {chain}: {emb.shape}")
                        matching_embeddings.append(emb)
                    else:
                        missing_chains.append(chain)
                
                if missing_chains:
                    raise ValueError(f"Missing embeddings for chains: {missing_chains[:10]}...")
                
                embeddings_list = matching_embeddings
            
            # Handle tensor format
            elif isinstance(embeddings, torch.Tensor):
                if embeddings.dim() == 3:  # [batch, seq_len, dim]
                    # Check if batch dimension matches expected count
                    if embeddings.size(0) != len(dataframe):
                        raise ValueError(
                            f"Mismatch: {embeddings.size(0)} embeddings vs {len(dataframe)} dataframe entries"
                        )
                    # Convert to global embeddings
                    embeddings_list = [embeddings[i].mean(dim=0) for i in range(embeddings.size(0))]
                elif embeddings.dim() == 2:  # [batch, dim] - Already global
                    if embeddings.size(0) != len(dataframe):
                        raise ValueError(
                            f"Mismatch: {embeddings.size(0)} embeddings vs {len(dataframe)} dataframe entries"
                        )
                    embeddings_list = [embeddings[i] for i in range(embeddings.size(0))]
                else:
                    raise ValueError(f"Unexpected tensor shape: {embeddings.shape}")
            else:
                # Assume list format
                embeddings_list = embeddings
            
            # Validate alignment for evaluation
            if len(embeddings_list) != len(labels):
                raise ValueError(f"Mismatch: {len(embeddings_list)} embeddings vs {len(labels)} labels")
            if len(embeddings_list) != len(dataframe):
                raise ValueError(f"Mismatch: {len(embeddings_list)} embeddings vs {len(dataframe)} dataframe rows")
            
            return embeddings_list, labels, dataframe

    def _ensure_global_embeddings(self, embeddings, processed_data):
        """Ensure embeddings are in global format for evaluation task"""
        if isinstance(embeddings, torch.Tensor):
            # Check tensor shape
            if embeddings.dim() == 3:
                print(f"Converting 3D embeddings tensor with shape {embeddings.shape} to global embeddings")
                # Mean pool across sequence dimension
                return embeddings.mean(dim=1)
            elif embeddings.dim() == 2:
                # Already global - one embedding per sequence
                return embeddings
        elif isinstance(embeddings, dict):
            # Get values and check first one
            sample = next(iter(embeddings.values()))
            if sample.dim() == 2:
                # Per-residue embeddings in dict
                return {k: v.mean(dim=0) for k, v in embeddings.items()}
        
        # Return unchanged if already in correct format
        return embeddings
    
    def prepare_dataloaders(self, *args):
        """Prepare appropriate dataloaders based on task type"""
        if self.task_type == "detection":
            embeddings, processed_data = args
            
            try:
                dataset = CoreDetectionDataset(embeddings, processed_data)
                print(f"Created CoreDetectionDataset with {len(dataset)} items")
            except Exception as e:
                print(f"ERROR creating CoreDetectionDataset: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
        else:  # evaluation
            embeddings, labels, dataframe = args
            
            try:
                dataset = SequenceDataset(embeddings, labels, dataframe)
                print(f"Created SequenceDataset with {len(dataset)} items")
            except Exception as e:
                print(f"ERROR creating SequenceDataset: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        # Verify dataset creation was successful
        assert len(dataset) > 0, "Created dataset is empty"
        
        # Get a sample item to verify structure
        try:
            sample_item = dataset[0]
            print(f"Sample item keys: {list(sample_item.keys())}")
            for k, v in sample_item.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        except Exception as e:
            print(f"ERROR accessing sample item: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Split dataset
        train_size = int(len(dataset) * self.config.train_ratio)
        val_size = len(dataset) - train_size
        
        print(f"Splitting dataset: train={train_size}, val={val_size}")
        assert train_size > 0, "Train dataset would be empty"
        assert val_size > 0, "Validation dataset would be empty"
        
        generator = torch.Generator().manual_seed(self.config.seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        # Create dataloaders with robust worker settings
        num_workers = getattr(self.config, 'num_workers', 0)  # Default to 0 for safety
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        print(f"Created DataLoaders: train={len(train_loader)} batches, val={len(val_loader)} batches")
        
        # Test first batch to ensure no errors
        try:
            train_batch = next(iter(train_loader))
            print(f"Successfully retrieved first batch")
            print(f"Batch keys: {list(train_batch.keys())}")
            print(f"Batch size: {train_batch['embeddings'].shape[0]}")
        except Exception as e:
            print(f"ERROR retrieving first batch: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        return train_loader, val_loader