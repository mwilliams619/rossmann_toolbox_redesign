class PDBSequenceExtractor:
    def __init__(self, pdb_dir):
        self.pdb_dir = Path(pdb_dir)
        self.parser = PDB.PDBParser(QUIET=True)
        
    def extract_sequence(self, pdb_id):
        """Extract sequence from a single PDB file"""
        pdb_file = self.pdb_dir / f"{pdb_id}.pdb"
        
        if not pdb_file.exists():
            return pdb_id, None
        
        try:
            structure = self.parser.get_structure(pdb_id, pdb_file)
            full_sequence = []
            
            # Extract sequence from each chain
            for model in structure:
                for chain in model:
                    residues = [res for res in chain if PDB.is_aa(res, standard=True)]
                    if residues:
                        sequence = ''
                        for res in residues:
                            try:
                                # Get one-letter code for amino acid
                                sequence += Polypeptide.three_to_one(res.get_resname())
                            except KeyError:
                                sequence += 'X'  # Unknown amino acid
                        full_sequence.append(sequence)
            
            # Combine all chain sequences
            return pdb_id, ''.join(full_sequence) if full_sequence else None
        
        except Exception as e:
            print(f"Error processing {pdb_id}: {str(e)}")
            return pdb_id, None
    
    def batch_extract_sequences(self, pdb_ids, num_workers=4):
        """Extract sequences from multiple PDB files in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_pdb = {
                executor.submit(self.extract_sequence, pdb_id): pdb_id 
                for pdb_id in pdb_ids
            }
            
            for future in tqdm(as_completed(future_to_pdb), total=len(pdb_ids)):
                pdb_id, sequence = future.result()
                results[pdb_id] = sequence
        
        return results