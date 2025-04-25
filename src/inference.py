import argparse
import sys
import torch
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json

# Import your model classes
from models.seq_detect import ModernSeqCoreEvaluator, ModernSeqCoreDetector


class InferenceEngine:
    def __init__(self, evaluator_path, detector_path, device=None):
        """Initialize inference engine with pretrained models"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load ESM-2 model for embedding generation
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(self.device)
        self.esm_model.eval()
        
        # Load pretrained evaluator with validation
        evaluator_checkpoint = torch.load(evaluator_path, map_location=self.device)
        print(f"Evaluator checkpoint keys: {list(evaluator_checkpoint.keys()) if isinstance(evaluator_checkpoint, dict) else 'direct state dict'}")
        
        # Get config if available
        config = evaluator_checkpoint.get('config', {}) if isinstance(evaluator_checkpoint, dict) else {}
        self.evaluator = ModernSeqCoreEvaluator(embedding_dim=config.get('embedding_dim', 1280)).to(self.device)
        
        # Handle different checkpoint formats with validation
        if isinstance(evaluator_checkpoint, dict) and 'model_state_dict' in evaluator_checkpoint:
            state_dict = evaluator_checkpoint['model_state_dict']
        else:
            state_dict = evaluator_checkpoint
        
        # Validate state dict keys match
        missing_keys, unexpected_keys = self.evaluator.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"WARNING: Missing keys in evaluator: {missing_keys}")
        if unexpected_keys:
            print(f"WARNING: Unexpected keys in evaluator: {unexpected_keys}")
        
        self.evaluator.eval()
        
        # Repeat the same pattern for detector
        detector_checkpoint = torch.load(detector_path, map_location=self.device)
        print(f"Detector checkpoint keys: {list(detector_checkpoint.keys()) if isinstance(detector_checkpoint, dict) else 'direct state dict'}")
        
        config = detector_checkpoint.get('config', {}) if isinstance(detector_checkpoint, dict) else {}
        self.detector = ModernSeqCoreDetector(embedding_dim=1280).to(self.device)
        
        if isinstance(detector_checkpoint, dict) and 'model_state_dict' in detector_checkpoint:
            state_dict = detector_checkpoint['model_state_dict']
        else:
            state_dict = detector_checkpoint
        
        missing_keys, unexpected_keys = self.detector.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"WARNING: Missing keys in detector: {missing_keys}")
        if unexpected_keys:
            print(f"WARNING: Unexpected keys in detector: {unexpected_keys}")
            
        self.detector.eval()
        
        # Cofactor mapping
        self.cofactor_labels = ['FAD', 'NAD', 'NADP', 'SAM']
        
        # Validate models work as expected
        self._validate_models()

    def _validate_models(self):
        """Check that models produce different outputs for different inputs"""
        with torch.no_grad():
            # Create two different random inputs
            rand1 = torch.randn(1, 10, 1280).to(self.device)
            rand2 = torch.randn(1, 10, 1280).to(self.device)
            
            # Test evaluator
            out1 = self.evaluator(rand1.mean(dim=1))
            out2 = self.evaluator(rand2.mean(dim=1))
            if torch.allclose(out1, out2, rtol=1e-3):
                print("WARNING: Evaluator produces identical outputs for different inputs!")
                print(f"Output 1: {out1.tolist()}")
                print(f"Output 2: {out2.tolist()}")
            else:
                print("Evaluator validation successful")
            
            # Test detector
            core1, res1 = self.detector(rand1)
            core2, res2 = self.detector(rand2)
            if torch.allclose(core1, core2, rtol=1e-3) and torch.allclose(res1, res2, rtol=1e-3):
                print("WARNING: Detector produces identical outputs for different inputs!")
            else:
                print("Detector validation successful")
    
    def generate_embedding(self, sequence):
        """Generate embeddings for a single sequence"""
        inputs = self.tokenizer(
            [sequence], 
            padding=True, 
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)
        
        print(f"Tokenized sequence length: {inputs['input_ids'].shape[1]}")
        
        with torch.no_grad():
            outputs = self.esm_model(**inputs, output_hidden_states=True)
            # Use both last hidden state and all hidden states
            last_hidden = outputs.last_hidden_state[0]  # [seq_len, 1280]
        
        # More careful token handling
        seq_len = len(sequence)
        if last_hidden.shape[0] > seq_len + 2:  # +2 for special tokens
            print(f"WARNING: Tokenized length ({last_hidden.shape[0]}) > sequence length ({seq_len})")
            
        # Extract embeddings for actual amino acids (skipping special tokens)
        embedding = last_hidden[1:seq_len+1]
        
        print(f"Final embedding shape: {embedding.shape}")
        return embedding
    
    def predict_cofactor(self, embedding):
        """Predict cofactor type using the evaluator model"""
        # Mean pool the per-residue embeddings for the evaluator
        global_emb = embedding.mean(dim=0).unsqueeze(0)  # [1, 1280]
        
        print(f"Input embedding stats: mean={global_emb.mean().item():.4f}, std={global_emb.std().item():.4f}")
        
        with torch.no_grad():
            logits = self.evaluator(global_emb)
            print(f"Raw logits: {logits.tolist()}")
            
            probs = torch.softmax(logits, dim=1)
            print(f"Probabilities: {probs.tolist()}")
            
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        return self.cofactor_labels[pred_idx], confidence, probs[0].cpu().numpy()
    
    def detect_core_residues(self, embedding):
        """Detect core region and critical residues"""
        with torch.no_grad():
            # Add batch dimension
            emb_batch = embedding.unsqueeze(0)  # [1, seq_len, 1280]
            
            # Get predictions
            core_scores, residue_scores = self.detector(emb_batch)
            
            # Process outputs
            core_probs = core_scores[0].cpu().numpy()
            residue_probs = residue_scores[0].cpu().numpy()
            
            # Identify core region
            core_mask = core_probs > 0.5
            core_start, core_end = self._find_core_boundaries(core_mask)
            
            # Identify critical residues
            critical_residues = []
            for i, prob in enumerate(residue_probs):
                if prob > 0.5:
                    critical_residues.append(i)
        
        return core_start, core_end, critical_residues, core_probs, residue_probs
    
    def _find_core_boundaries(self, core_mask):
        """Find continuous core region boundaries"""
        if not any(core_mask):
            return None, None
        
        # Find the first and last True values
        core_indices = torch.nonzero(torch.tensor(core_mask)).squeeze()
        if len(core_indices.shape) == 0:  # Single value
            core_indices = core_indices.unsqueeze(0)
        
        if len(core_indices) > 0:
            return core_indices[0].item(), core_indices[-1].item() + 1
        return None, None
    
    def process_sequence(self, sequence, seq_id=None):
        """Process a single sequence and return predictions"""
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Invalid sequence")
        
        # Generate embedding
        embedding = self.generate_embedding(sequence)
        
        # Predict cofactor
        cofactor, confidence, all_probs = self.predict_cofactor(embedding)
        
        # Detect core and residues
        core_start, core_end, critical_residues, core_probs, residue_probs = \
            self.detect_core_residues(embedding)
        
        # Extract core sequence
        core_seq = sequence[core_start:core_end] if core_start is not None else ""
        
        # Create result dictionary
        result = {
            'sequence_id': seq_id or 'unnamed',
            'sequence': sequence,
            'cofactor': cofactor,
            'cofactor_confidence': f"{confidence:.3f}",
            'all_cofactor_probs': {label: f"{prob:.3f}" for label, prob in zip(self.cofactor_labels, all_probs)},
            'core_start': core_start,
            'core_end': core_end,
            'core_sequence': core_seq,
            'critical_residues': critical_residues,
            'critical_residue_positions': [i+1 for i in critical_residues]  # 1-indexed
        }
        
        return result
    
    def process_fasta(self, fasta_path, batch_size=16):
        """Process multiple sequences from a FASTA file"""
        results = []
        sequences = list(SeqIO.parse(fasta_path, "fasta"))
        
        print(f"Processing {len(sequences)} sequences...")
        
        for seq_record in tqdm(sequences):
            try:
                result = self.process_sequence(str(seq_record.seq), seq_record.id)
                results.append(result)
            except Exception as e:
                print(f"Error processing {seq_record.id}: {str(e)}")
                continue
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Protein sequence prediction using pretrained models")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sequence', type=str, help="Single protein sequence")
    input_group.add_argument('--fasta', type=str, help="FASTA file containing multiple sequences")
    
    parser.add_argument('--evaluator', type=str, required=True, help="Path to pretrained evaluator model")
    parser.add_argument('--detector', type=str, required=True, help="Path to pretrained detector model")
    parser.add_argument('--output', type=str, default=None, help="Output file (JSON or CSV)")
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json', 
                        help="Output format")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = InferenceEngine(args.evaluator, args.detector, args.device)
    
    # Process input
    if args.sequence:
        results = [engine.process_sequence(args.sequence, seq_id="input_sequence")]
    else:
        results = engine.process_fasta(args.fasta)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        
        if args.format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:  # CSV
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        
        print(f"Results saved to {output_path}")
    else:
        # Print to console
        for result in results:
            print("\n" + "="*50)
            print(f"Sequence ID: {result['sequence_id']}")
            print(f"Cofactor prediction: {result['cofactor']} (confidence: {result['cofactor_confidence']})")
            print(f"Core region: {result['core_start']}-{result['core_end']}")
            print(f"Core sequence: {result['core_sequence']}")
            print(f"Critical residues: {result['critical_residue_positions']}")
            print(f"All cofactor probabilities: {result['all_cofactor_probs']}")
    
    # Create a simple visualization if requested
    if len(results) > 0 and not args.output:
        result = results[0]
        sequence = result['sequence']
        core_start = result['core_start']
        core_end = result['core_end']
        critical_residues = result['critical_residues']
        
        print("\nSequence visualization:")
        for i, residue in enumerate(sequence):
            if core_start is not None and core_start <= i < core_end:
                if i in critical_residues:
                    print(f"\033[91m{residue}\033[0m", end="")  # Red for critical
                else:
                    print(f"\033[92m{residue}\033[0m", end="")  # Green for core
            else:
                print(residue, end="")
        print("\n")
        print("Legend: \033[92mCore region\033[0m, \033[91mCritical residues\033[0m")


if __name__ == "__main__":
    main()