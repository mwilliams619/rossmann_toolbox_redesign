from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from transformers import pipeline

class ProteinEmbedding:
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D", device='cuda'):
        self.device = device
        
        try:
            # First try loading tokenizer and model separately
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            # If above succeeds, create the pipeline
            from transformers import pipeline
            self.pipe = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, device=device)
            
        except Exception as e:
            print(f"Error loading model separately: {e}")
            print("Trying alternate approach with pipeline...")
            
            # Fallback: try loading directly with pipeline
            from transformers import pipeline
            self.pipe = pipeline("fill-mask", model=model_name, device=device)
            self.tokenizer = self.pipe.tokenizer
            self.model = self.pipe.model

    @torch.no_grad()
    def generate_embeddings(self, sequences, batch_size=32):
        """Generate embeddings for a list of sequences"""
        embeddings = []

        if len(sequences) > 0:
                sample_inputs = self.tokenizer([sequences[0]], return_tensors="pt", padding=True)
                sample_inputs = {k: v.to(self.device) for k, v in sample_inputs.items()}
                with torch.no_grad():
                    sample_outputs = self.model(**sample_inputs)
                
                print("Model output structure:")
                print(f"Type: {type(sample_outputs)}")
                print(f"Available attributes: {dir(sample_outputs)}")
                if isinstance(sample_outputs, tuple):
                    print(f"Tuple length: {len(sample_outputs)}")
                    for i, item in enumerate(sample_outputs):
                        print(f"Item {i} type: {type(item)}")
                        if hasattr(item, 'keys'):
                            print(f"Item {i} keys: {item.keys()}")
        
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_sequences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=65
            ).to(self.device)
            
            outputs = self.model(**inputs, output_hidden_states=True)

            # Get the last hidden state (actual embeddings)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[-1]
            
            print(f"Outputs: {outputs}")
            print(f"Outputs type: {type(outputs)}")
            print(f"Outputs keys: {outputs.keys()}")
            print(f"Outputs last_hidden_state shape: {hidden_states.shape}")
            print(f"Outputs last_hidden_state type: {type(hidden_states)}")
            
            # Mean pooling over sequence length
            batch_embeddings = hidden_states.mean(dim=1)
            embeddings.append(batch_embeddings.cpu())
            
        return torch.cat(embeddings, dim=0).numpy()

    def save_embeddings(self, embeddings, filepath):
        """Save embeddings to file"""
        torch.save(embeddings, filepath)

    def load_embeddings(self, filepath):
        """Load embeddings from file"""
        return torch.load(filepath)