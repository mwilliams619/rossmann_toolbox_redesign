import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoModel
import ast

# Import your model classes
from models.seq_detect import ModernSeqCoreEvaluator, ModernSeqCoreDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

class SimplifiedDiagnostics:
    def __init__(self, evaluator_path, detector_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Define possible cofactor mappings to test
        self.possible_mappings = [
            ['NAD', 'NADP', 'FAD', 'SAM'],  # Hypothesis based on confusion matrix
            ['NADP', 'NAD', 'FAD', 'SAM'],  # Swapped NAD/NADP
            ['FAD', 'NAD', 'NADP', 'SAM'],  # Different order
            ['SAM', 'NAD', 'NADP', 'FAD']   # Completely different
        ]
        
        # Load models
        self.load_models(evaluator_path, detector_path)
        
        # Load ESM model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(self.device)
        self.esm_model.eval()
    
    def load_models(self, evaluator_path, detector_path):
        """Load models with proper configuration"""
        # Load evaluator
        self.evaluator = ModernSeqCoreEvaluator(embedding_dim=1280).to(self.device)
        evaluator_checkpoint = torch.load(evaluator_path, map_location=self.device)
        
        # Get state dict from checkpoint
        if isinstance(evaluator_checkpoint, dict) and 'model_state_dict' in evaluator_checkpoint:
            state_dict = evaluator_checkpoint['model_state_dict']
        else:
            state_dict = evaluator_checkpoint
        
        self.evaluator.load_state_dict(state_dict, strict=False)
        self.evaluator.eval()
        
        # Patch forward method to return logits (remove softmax)
        original_forward = self.evaluator.forward
        def modified_forward(self, x):
            if len(x.shape) == 2:  # (batch_size, embedding_dim)
                x = x.unsqueeze(1)  # (batch_size, 1, embedding_dim)
            x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_length)
            x = self.input_proj(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.attention(x)
            x = self.pool(x).squeeze(-1)
            x = self.norm(x)
            x = self.classifier(x)
            return x  # Return raw logits
        self.evaluator.forward = modified_forward.__get__(self.evaluator, ModernSeqCoreEvaluator)
        
        # Load detector
        self.detector = ModernSeqCoreDetector(embedding_dim=1280).to(self.device)
        detector_checkpoint = torch.load(detector_path, map_location=self.device)
        
        if isinstance(detector_checkpoint, dict) and 'model_state_dict' in detector_checkpoint:
            state_dict = detector_checkpoint['model_state_dict']
        else:
            state_dict = detector_checkpoint
        
        self.detector.load_state_dict(state_dict, strict=False)
        self.detector.eval()
        
        logging.info("Models loaded successfully")
    
    def parse_csv(self, csv_path):
        """Load and parse CSV, handling special formats and cleaning data"""
        logging.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} rows from CSV")
        
        # Standardize cofactor labels
        cofactor_mapping = {
            'NAD': 'NAD', 'NADH': 'NAD', 
            'NADP': 'NADP', 'NADPH': 'NADP', 'NDP': 'NADP', 'NAP': 'NADP',
            'FAD': 'FAD', 'FADH': 'FAD', 'FAD2': 'FAD',
            'SAM': 'SAM', 'SAH': 'SAM'
        }
        
        # Check which column to use for cofactor
        if 'simplified_cofactor' in df.columns:
            df['standard_cofactor'] = df['simplified_cofactor'].str.strip().str.upper().map(cofactor_mapping)
        elif 'cofactor' in df.columns:
            df['standard_cofactor'] = df['cofactor'].str.strip().str.upper().map(cofactor_mapping)
        else:
            logging.error("No cofactor column found in CSV")
            return None
        
        # Log distribution
        cofactor_counts = df['standard_cofactor'].value_counts()
        logging.info(f"Cofactor distribution: {cofactor_counts.to_dict()}")
        
        # Verify we have the right cofactors
        expected_cofactors = set(['NAD', 'NADP', 'FAD', 'SAM'])
        found_cofactors = set(df['standard_cofactor'].unique())
        if not expected_cofactors.issubset(found_cofactors):
            missing = expected_cofactors - found_cofactors
            logging.warning(f"Missing cofactors in data: {missing}")
        
        # Get valid sequences
        valid_df = df.dropna(subset=['seq', 'standard_cofactor']).reset_index(drop=True)
        logging.info(f"Found {len(valid_df)} valid sequences with cofactors")
        
        return valid_df
    
    def generate_embedding(self, sequence):
        """Generate embeddings for sequence"""
        inputs = self.tokenizer(
            [sequence], 
            padding=True, 
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            embedding = outputs.last_hidden_state[0]  # [seq_len, 1280]
        
        # Remove special tokens
        seq_len = len(sequence)
        if embedding.shape[0] >= 2:
            embedding = embedding[1:seq_len+1]  # Skip start token
        
        return embedding
    
    def test_mappings(self, df, sample_size=30):
        """Test all mappings on a sample of sequences to find the best one"""
        # Sample data for evaluation
        if len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
        
        results = []
        
        for mapping in self.possible_mappings:
            correct_count = 0
            total = len(sample_df)
            pbar = tqdm(sample_df.iterrows(), total=total, desc=f"Testing mapping {mapping}")
            
            for idx, row in pbar:
                sequence = row['seq']
                true_cofactor = row['standard_cofactor']
                
                # Generate embedding
                embedding = self.generate_embedding(sequence)
                global_emb = embedding.mean(dim=0).unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    logits = self.evaluator(global_emb)
                    probs = F.softmax(logits, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    pred_cofactor = mapping[pred_idx]
                
                if pred_cofactor == true_cofactor:
                    correct_count += 1
                
                pbar.set_postfix({'correct': correct_count, 'accuracy': correct_count/total})
            
            accuracy = correct_count / total
            logging.info(f"Mapping {mapping} accuracy: {accuracy:.4f} ({correct_count}/{total} correct)")
            
            results.append({
                'mapping': mapping,
                'accuracy': accuracy,
                'correct': correct_count,
                'total': total
            })
        
        # Find best mapping
        best_result = max(results, key=lambda x: x['accuracy'])
        logging.info(f"\nBEST MAPPING: {best_result['mapping']}")
        logging.info(f"Accuracy: {best_result['accuracy']:.4f} ({best_result['correct']}/{best_result['total']} correct)")
        
        return results
    
    def verify_with_single_sequence(self, sequence, true_cofactor=None):
        """Verify model behavior with a single sequence"""
        logging.info(f"Testing sequence: {sequence[:50]}...")
        
        # Generate embedding
        embedding = self.generate_embedding(sequence)
        global_emb = embedding.mean(dim=0).unsqueeze(0)
        
        # Get raw outputs
        with torch.no_grad():
            logits = self.evaluator(global_emb)
            logging.info(f"Raw logits: {logits.tolist()}")
            
            probs = F.softmax(logits, dim=1)[0]
            logging.info(f"Softmax probabilities: {probs.tolist()}")
            
            # Show predictions with all mappings
            for mapping in self.possible_mappings:
                pred_idx = torch.argmax(probs).item()
                pred_cofactor = mapping[pred_idx]
                confidence = probs[pred_idx].item()
                
                logging.info(f"With mapping {mapping}: {pred_cofactor} (confidence: {confidence:.4f})")
                if true_cofactor and pred_cofactor == true_cofactor:
                    logging.info(f"  --> MATCH with true cofactor {true_cofactor}")
        
        # Test detector
        with torch.no_grad():
            core_emb = embedding.unsqueeze(0)
            core_scores, residue_scores = self.detector(core_emb)
            core_mask = (core_scores[0] > 0.5).cpu().numpy()
            core_indices = np.where(core_mask)[0]
            
            if len(core_indices) > 0:
                core_start = core_indices[0]
                core_end = core_indices[-1] + 1
                core_seq = sequence[core_start:core_end]
                logging.info(f"Detected core: {core_start}-{core_end}: {core_seq}")
            else:
                logging.info("No core region detected")

def main():
    parser = argparse.ArgumentParser(description="Simplified diagnostic for model evaluation")
    parser.add_argument('--evaluator', required=True, help="Path to evaluator model checkpoint")
    parser.add_argument('--detector', required=True, help="Path to detector model checkpoint")
    parser.add_argument('--csv', required=True, help="Path to training data CSV")
    parser.add_argument('--sample-size', type=int, default=30, help="Number of samples to test")
    parser.add_argument('--test-sequence', type=str, help="Single sequence to test")
    parser.add_argument('--true-cofactor', type=str, help="True cofactor for test sequence")
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None, help="Device for computation")
    
    args = parser.parse_args()
    
    # Initialize diagnostics
    diagnostics = SimplifiedDiagnostics(args.evaluator, args.detector, args.device)
    
    # Parse CSV
    df = diagnostics.parse_csv(args.csv)
    if df is None:
        return
    
    # Test mappings
    results = diagnostics.test_mappings(df, args.sample_size)
    
    # Test single sequence if provided
    if args.test_sequence:
        diagnostics.verify_with_single_sequence(args.test_sequence, args.true_cofactor)
        
    # Create visualization of results
    plt.figure(figsize=(10, 6))
    mappings = [r['mapping'][0] + '/' + r['mapping'][1] + '/...' for r in results]
    accuracies = [r['accuracy'] for r in results]
    bars = plt.bar(mappings, accuracies)
    
    # Highlight best mapping
    best_idx = accuracies.index(max(accuracies))
    bars[best_idx].set_color('green')
    
    plt.xlabel('Mapping (first two classes shown)')
    plt.ylabel('Accuracy')
    plt.title('Cofactor Mapping Accuracy')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('mapping_results.png')
    logging.info("Results chart saved as 'mapping_results.png'")

if __name__ == "__main__":
    main()