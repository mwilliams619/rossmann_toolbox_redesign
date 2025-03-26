from pathlib import Path
from data.preprocessing import RossmannDataProcessor
from utils.embeddings import ProteinEmbedding
import pickle

def main():
    # Initialize processor
    data_dir = Path("data")
    processor = RossmannDataProcessor(data_dir)
    
    # Load pickle data
    raw_data = processor.load_pickle_data("cofdata_core_v12_20200717.p")
    
    # Process data
    processed_data = processor.prepare_dataset(raw_data)
    
    # Generate embeddings
    embedding_generator = ProteinEmbedding()
    embeddings = embedding_generator.generate_embeddings(processed_data['sequences'])
    
    # Save processed data and embeddings
    embedding_generator.save_embeddings(
        embeddings, 
        data_dir / "processed" / "embeddings.pt"
    )
    
    # Save processed data
    with open(data_dir / "processed" / "processed_data.pkl", 'wb') as f:
        pickle.dump(processed_data, f)

if __name__ == "__main__":
    main()