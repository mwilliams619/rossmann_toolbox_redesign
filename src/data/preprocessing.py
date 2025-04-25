import torch
from pathlib import Path
import pickle
from Bio import PDB
from Bio.PDB import PPBuilder
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from collections import Counter
import os

class SimpleFullSequenceProcessor:
    def __init__(self, data_dir: Path, pdb_dir: Path):
        self.data_dir = Path(data_dir)
        self.pdb_dir = Path(pdb_dir)
        self.processed_dir = 'data/processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        # Error tracking
        self.stats = {
            'empty_files': 0,
            'missing_files': 0,
            'parsed_files': 0
        }
        
        # Cofactor mapping
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
    
    def extract_full_sequence(self, pdb_filename):
        """Extract the FULL sequence from a PDB file, ignoring chains"""
        # Resolve the file path
        file_path = self.pdb_dir / pdb_filename
        
        # Check if file exists
        if not file_path.exists():
            self.stats['missing_files'] += 1
            return None
        
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            self.stats['empty_files'] += 1
            return None
        
        try:
            # Parse the PDB file
            parser = PDB.PDBParser(QUIET=True, PERMISSIVE=True)
            structure = parser.get_structure("structure", file_path)
            
            # Extract sequences from all chains and join them
            ppb = PPBuilder()
            all_peptides = []
            
            for chain in structure.get_chains():
                peptides = list(ppb.build_peptides(chain))
                all_peptides.extend(peptides)
            
            if all_peptides:
                # Join all peptide sequences to get the full sequence
                full_sequence = ''.join(str(peptide.get_sequence()) for peptide in all_peptides)
                self.stats['parsed_files'] += 1
                return full_sequence
            else:
                return None
                
        except Exception as e:
            # print(f"Error extracting sequence from {pdb_filename}: {e}")
            return None
    
    def process_dataset(self, data_path):
        """Process the dataset and extract full sequences"""
        # Load the dataset
        if str(data_path).endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            with open(data_path, 'rb') as f:
                data = pd.read_pickle(f)
        
        print(f"Loaded dataset with {len(data)} rows")
        
        # Check if core_pdb column exists
        if 'core_pdb' not in data.columns:
            print("Error: 'core_pdb' column not found")
            return None
            
        # Get unique core PDB values
        core_pdbs = data['core_pdb'].dropna().unique()
        print(f"Found {len(core_pdbs)} unique PDB files to process")
        
        # Process files in parallel
        full_sequences = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.extract_full_sequence, pdb_file): pdb_file 
                      for pdb_file in core_pdbs}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting full sequences"):
                pdb_file = futures[future]
                try:
                    result = future.result()
                    if result:
                        full_sequences[pdb_file] = result
                except Exception as e:
                    pass
        
        print("\nFile processing statistics:")
        print(f"Total PDB files: {len(core_pdbs)}")
        print(f"Successfully parsed: {self.stats['parsed_files']} ({self.stats['parsed_files']/len(core_pdbs)*100:.2f}%)")
        print(f"Empty files: {self.stats['empty_files']} ({self.stats['empty_files']/len(core_pdbs)*100:.2f}%)")
        print(f"Missing files: {self.stats['missing_files']} ({self.stats['missing_files']/len(core_pdbs)*100:.2f}%)")
        
        print("\nProcessing dataset rows...")
        processed_rows = []
        skipped_reasons = Counter()
        
        # Process rows
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
            # Skip if no core_pdb
            if pd.isna(row['core_pdb']):
                skipped_reasons['missing_core_pdb'] += 1
                continue
                
            pdb_chain = row['pdb_chain']
            core_pdb = row['core_pdb']
            
            # Skip if no full sequence for this file
            if core_pdb not in full_sequences:
                skipped_reasons['file_not_processed'] += 1
                continue
            
            # Get the full sequence
            full_seq = full_sequences[core_pdb]
            
            try:
                # Get core info from the row data - we trust this info
                core_seq = row['seq']
                core_start = int(row['seqres_start'])
                core_end = int(row['seqres_end'])
                
                # Process SS positions
                ss_pos = row['ss_pos']
                if isinstance(ss_pos, str):
                    # Try various formats
                    try:
                        if 'np.int64' in ss_pos:
                            # Format: "[np.int64(6), np.int64(12),...]"
                            clean_str = ss_pos.replace('np.int64', '').replace('(', '').replace(')', '')
                            ss_pos = eval(clean_str)
                        else:
                            # Try direct eval
                            ss_pos = eval(ss_pos)
                    except:
                        skipped_reasons['invalid_ss_pos'] += 1
                        continue
                        
                # Adjust ss_pos to full sequence
                try:
                    adjusted_ss_pos = [pos + core_start for pos in ss_pos]
                except TypeError:
                    skipped_reasons['ss_pos_not_iterable'] += 1
                    continue
                
                # Add to processed rows
                processed_rows.append({
                    'pdb_chain': pdb_chain,
                    'full_seq': full_seq,
                    'core_seq': core_seq,
                    'core_start': core_start,
                    'core_end': core_end,
                    'ss_pos_full': adjusted_ss_pos,
                    'ss_pos_core': ss_pos,
                    'cofactor': row['simplified_cofactor'],
                    'original_idx': idx
                })
                
            except Exception as e:
                skipped_reasons[f'exception:{str(e)[:50]}'] += 1
                continue
        
        # Create processed DataFrame
        processed_df = pd.DataFrame(processed_rows)
        print(f"Final processed dataset: {len(processed_df)} rows")
        
        # Print skipped reasons
        print("\nRows skipped by reason:")
        for reason, count in skipped_reasons.most_common():
            print(f"  {reason}: {count} ({count/len(data)*100:.2f}%)")
        
        if len(processed_df) == 0:
            print("Error: No rows after processing!")
            return None
        
        # Get maximum sequence length
        max_seq_len = max(len(seq) for seq in processed_df['full_seq'])
        print(f"Maximum sequence length: {max_seq_len}")
        
        # Create core and residue labels
        core_labels = np.zeros((len(processed_df), max_seq_len), dtype=np.float32)
        residue_labels = np.zeros((len(processed_df), max_seq_len), dtype=np.float32)
        
        for i, row in processed_df.iterrows():
            # Core labels - cap at max_seq_len
            start = min(row['core_start'], max_seq_len-1)
            end = min(row['core_end'], max_seq_len)
            if start < end:
                core_labels[i, start:end] = 1.0
            
            # Residue labels
            for pos in row['ss_pos_full']:
                if 0 <= pos < max_seq_len:
                    residue_labels[i, pos] = 1.0
        
        # Create cofactor labels
        cofactors = processed_df['cofactor'].tolist()
        unique_cofactors = sorted(set(self.cofactor_groups.keys()))
        cofactor_to_idx = {cof: idx for idx, cof in enumerate(unique_cofactors)}
        
        cofactor_labels = np.zeros((len(cofactors), len(unique_cofactors)), dtype=np.float32)
        for i, cof in enumerate(cofactors):
            mapped_cof = self.cofactor_to_group.get(cof, cof)
            if mapped_cof in cofactor_to_idx:
                cofactor_labels[i, cofactor_to_idx[mapped_cof]] = 1.0
        
        # Print cofactor distribution
        cofactor_counts = processed_df['cofactor'].value_counts()
        print("\nCofactor distribution:")
        for cof, count in cofactor_counts.items():
            print(f"  {cof}: {count} ({count/len(processed_df)*100:.2f}%)")
        
        # Package data
        processed_data = {
            'labels': cofactor_labels,
            'core_labels': core_labels,
            'residue_labels': residue_labels,
            'dataframe': processed_df,
            'cofactor_mapping': {idx: cof for cof, idx in cofactor_to_idx.items()}
        }
        
        # Extract sequences for embedding generation
        full_sequences_list = processed_df['full_seq'].tolist()
        chains_list = processed_df['pdb_chain'].tolist()
        
        return processed_data, full_sequences_list, chains_list
    
    def generate_embeddings(self, sequences, chains_list, output_path):
        """Generate embeddings for sequences using ESM-2"""
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "facebook/esm2_t33_650M_UR50D"
        print(f"Generating embeddings for {len(sequences)} sequences using {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = model.to(device)
        model.eval()
        
        # Process in batches
        batch_size = 8
        all_embeddings = {}
        errors = {}
        
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i+batch_size]
            batch_chains = chains_list[i:i+batch_size]
            
            try:
                # Tokenize sequences
                inputs = tokenizer(
                    batch_sequences, 
                    padding=True, 
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt"
                ).to(device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.cpu()
                
                # Process each sequence in the batch
                for j, chain in enumerate(batch_chains):
                    seq_len = len(batch_sequences[j])
                    
                    # Extract actual sequence embeddings (exclude special tokens)
                    if seq_len <= 1022:  # ESM uses <cls> and <eos> tokens
                        embedding = embeddings[j, 1:seq_len+1]  # Skip <cls> token
                    else:
                        # Handle case where truncation happened
                        embedding = embeddings[j, 1:-1]  # Skip <cls> and <eos> tokens
                    
                    # Store embedding
                    all_embeddings[chain] = embedding
                    
            except Exception as e:
                for chain in batch_chains:
                    errors[chain] = str(e)
                print(f"Error processing batch starting at index {i}: {e}")
        
        # Save embeddings
        torch.save({
            'embeddings': all_embeddings,
            'metadata': {
                'format': 'per_sequence_dict',
                'num_sequences': len(all_embeddings),
                'num_errors': len(errors),
                'model': model_name,
                'embedding_dim': next(iter(all_embeddings.values())).shape[-1] if all_embeddings else None
            }
        }, output_path)
        
        print(f"Saved embeddings for {len(all_embeddings)} sequences to {output_path}")
        if errors:
            print(f"Encountered errors for {len(errors)} sequences")
        
        return all_embeddings
    
    def process_and_save(self, data_path):
        """Main processing function"""
        # Process the dataset
        result = self.process_dataset(data_path)
        
        if result is None:
            print("Processing failed")
            return
            
        processed_data, full_sequences, chains_list = result
        
        # Save processed data
        output_path = self.processed_dir / 'rossmann_full_seq_processed.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"Saved processed data to {output_path}")
        
        # Generate and save embeddings
        embeddings_path = self.processed_dir / 'rossmann_full_seq_embeddings.pt'
        self.generate_embeddings(full_sequences, chains_list, embeddings_path)
        
        print("\nProcessing complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python process_full_sequences.py <data_path> <pdb_dir>")
        sys.exit(1)
        
    data_path = sys.argv[1]
    pdb_dir = sys.argv[2]
    
    print(f"Data path: {data_path}")
    print(f"PDB directory: {pdb_dir}")
    
    processor = SimpleFullSequenceProcessor(Path(data_path).parent, Path(pdb_dir))
    processor.process_and_save(data_path)