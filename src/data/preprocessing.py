import torch
from pathlib import Path
import pickle
from Bio import SeqIO
import numpy as np
import pandas as pd

class RossmannDataProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        # Cofactor mapping dictionaries
        self.cofactor_groups = {
            'NAD': ['NAI', 'NAJ', 'NAD'],
            'NADP': ['NAP', 'NDP'],
            'FAD': ['FAD', 'FDA'],
            'SAM': ['SAM', 'SAH']
        }
        
        # Create reverse mapping
        self.cofactor_to_group = {}
        for group, cofactors in self.cofactor_groups.items():
            for cofactor in cofactors:
                self.cofactor_to_group[cofactor] = group

    def load_pickle_data(self, filename: str):
        """Load data from pickle file"""
        pickle_path = self.data_dir / 'raw' / filename
        with open(pickle_path, 'rb') as f:
            data1 = pd.read_pickle(f)
            # data = pickle.load(f)
        return data1

    def process_sequences(self, sequences, max_length=65):
        """Process and pad sequences to fixed length"""
        processed_seqs = []
        for seq in sequences:
            # Truncate if longer than max_length
            if len(seq) > max_length:
                seq = seq[:max_length]
            # Pad if shorter than max_length
            else:
                seq = seq + '-' * (max_length - len(seq))
            processed_seqs.append(seq)
        return processed_seqs

    def map_cofactors(self, cofactors):
        """Map detailed cofactor names to group names"""
        return [self.cofactor_to_group.get(cof, cof) for cof in cofactors]

    def prepare_dataset(self, data):
        """Prepare dataset for training"""
        sequences = self.process_sequences(data['seq'])
        cofactors = self.map_cofactors(data['simplified_cofactor'])
        
        # Create one-hot encoded labels
        unique_cofactors = sorted(set(self.cofactor_groups.keys()))
        cofactor_to_idx = {cof: idx for idx, cof in enumerate(unique_cofactors)}
        
        labels = np.zeros((len(cofactors), len(unique_cofactors)))
        for i, cof in enumerate(cofactors):
            labels[i, cofactor_to_idx[cof]] = 1
            
        return {
            'sequences': sequences,
            'labels': labels,
            'cofactor_mapping': cofactor_to_idx
        }