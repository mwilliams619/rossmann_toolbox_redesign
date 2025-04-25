import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
import pickle
import json
import os

# Import your inference engine and models
from inference import InferenceEngine
from models.seq_detect import ModernSeqCoreEvaluator, ModernSeqCoreDetector

def augment_dataset_from_fasta(
    fasta_path, 
    evaluator_path, 
    detector_path, 
    output_dir,
    device=None
):
    """Process a FASTA file to create a training dataset in the same format as the original data"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize inference engine
    print("Initializing inference engine...")
    engine = InferenceEngine(evaluator_path, detector_path, device)
    
    # Read sequences from FASTA
    sequences = list(SeqIO.parse(fasta_path, "fasta"))
    print(f"Processing {len(sequences)} sequences...")
    
    # Prepare data structures for processed data
    processed_rows = []
    max_seq_len = 0
    
    # Process each sequence
    for seq_record in tqdm(sequences, desc="Processing sequences"):
        sequence = str(seq_record.seq)
        seq_id = seq_record.id
        
        # Track maximum sequence length for padding
        max_seq_len = max(max_seq_len, len(sequence))
        
        try:
            # Process sequence using inference engine
            result = engine.process_sequence(sequence, seq_id)
            
            # Extract core region and critical residues
            core_start = result['core_start']
            core_end = result['core_end']
            core_seq = result['core_sequence']
            critical_residues = result['critical_residues']
            cofactor = result['cofactor']
            
            # Skip sequences where core wasn't detected
            if core_start is None or core_end is None:
                print(f"Skipping {seq_id}: No core region detected")
                continue
            
            # Adjust critical residue positions to be relative to core
            ss_pos_core = [pos - core_start for pos in critical_residues if core_start <= pos < core_end]
            
            # Create row similar to your original dataset format
            processed_rows.append({
                'pdb_chain': seq_id,  # Using sequence ID as a stand-in for pdb_chain
                'full_seq': sequence,
                'core_seq': core_seq,
                'core_start': core_start,
                'core_end': core_end,
                'ss_pos_full': critical_residues,
                'ss_pos_core': ss_pos_core,
                'cofactor': cofactor,
                'original_idx': len(processed_rows)
            })
            
        except Exception as e:
            print(f"Error processing {seq_id}: {str(e)}")
            continue
    
    # Create DataFrame
    print(f"Creating DataFrame with {len(processed_rows)} processed sequences")
    processed_df = pd.DataFrame(processed_rows)
    
    if len(processed_df) == 0:
        print("No sequences could be processed!")
        return None
    
    # Create labels arrays
    # Cofactor labels
    cofactor_groups = ['FAD', 'NAD', 'NADP', 'SAM']
    cofactor_to_idx = {cof: idx for idx, cof in enumerate(cofactor_groups)}
    
    cofactor_labels = np.zeros((len(processed_df), len(cofactor_groups)), dtype=np.float32)
    for i, cof in enumerate(processed_df['cofactor']):
        if cof in cofactor_to_idx:
            cofactor_labels[i, cofactor_to_idx[cof]] = 1.0
    
    # Core region labels
    core_labels = np.zeros((len(processed_df), max_seq_len), dtype=np.float32)
    residue_labels = np.zeros((len(processed_df), max_seq_len), dtype=np.float32)
    
    for i, row in processed_df.iterrows():
        # Set core region labels
        start = row['core_start']
        end = row['core_end']
        if 0 <= start < end <= max_seq_len:
            core_labels[i, start:end] = 1.0
        
        # Set critical residue labels
        for pos in row['ss_pos_full']:
            if 0 <= pos < max_seq_len:
                residue_labels[i, pos] = 1.0
    
    # Print cofactor distribution
    cofactor_counts = processed_df['cofactor'].value_counts()
    print("\nCofactor distribution:")
    for cof, count in cofactor_counts.items():
        print(f"  {cof}: {count} ({count/len(processed_df)*100:.2f}%)")
    
    # Package data in same format as original processing
    processed_data = {
        'labels': cofactor_labels,
        'core_labels': core_labels,
        'residue_labels': residue_labels,
        'dataframe': processed_df,
        'cofactor_mapping': {idx: cof for cof, idx in cofactor_to_idx.items()}
    }
    
    # Save processed data
    output_path = output_dir / 'augmented_data_processed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Saved processed data to {output_path}")
    
    # Save DataFrame as CSV for easier inspection
    csv_path = output_dir / 'augmented_data.csv'
    processed_df.to_csv(csv_path, index=False)
    print(f"Saved CSV version to {csv_path}")
    
    # Extract sequences for embedding generation (if needed)
    full_sequences_list = processed_df['full_seq'].tolist()
    chains_list = processed_df['pdb_chain'].tolist()
    
    return processed_data, full_sequences_list, chains_list

def main():
    parser = argparse.ArgumentParser(description="Create augmented dataset from FASTA file using pretrained models")
    
    parser.add_argument('--fasta', type=str,
                        help="FASTA file containing sequences", default='data/raw/cl0063_pfam.fasta')
    parser.add_argument('--evaluator', type=str, 
                        help="Path to pretrained evaluator model", default='model_outputs/evaluator/best_evaluator_model.pth')
    parser.add_argument('--detector', type=str,
                        help="Path to pretrained detector model", default='model_outputs/detector/best_detector_model.pth')
    parser.add_argument('--output', type=str, default='data/processed', 
                        help="Output directory for processed data")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help="Device to use for inference")
    parser.add_argument('--generate-embeddings', action='store_true',
                        help="Whether to generate embeddings for the sequences")
    
    args = parser.parse_args()
    
    # Process FASTA file and create dataset
    result = augment_dataset_from_fasta(
        args.fasta, 
        args.evaluator, 
        args.detector, 
        args.output,
        args.device
    )
    
    if result is None:
        print("Processing failed!")
        return
    
    processed_data, full_sequences, chains_list = result
    
    # Generate embeddings if requested
    if args.generate_embeddings:
        from src.data.preprocessing import SimpleFullSequenceProcessor  # Import your embedding generator

        # Create processor just for generating embeddings
        processor = SimpleFullSequenceProcessor(Path(args.output), Path(args.output))
        
        embeddings_path = Path(args.output) / 'augmented_embeddings.pt'
        processor.generate_embeddings(full_sequences, chains_list, embeddings_path)
        
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()