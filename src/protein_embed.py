import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import os
from tqdm import tqdm
import struct
import numpy as np

def load_fasta(fasta_file):
    sequences = []
    ids = []
    with open(fasta_file, 'r') as f:
        seq_buffer = []
        id_buffer = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if id_buffer:
                    ids.append(id_buffer)
                    sequences.append("".join(seq_buffer))
                id_buffer = line[1:]
                seq_buffer = []
            else:
                seq_buffer.append(line)
        if id_buffer:
            ids.append(id_buffer)
            sequences.append("".join(seq_buffer))
    return ids, sequences

def generate_embeddings(model_name, fasta_file, output_dir, batch_size=1):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    ids, sequences = load_fasta(fasta_file)
    print(f"Loaded {len(sequences)} sequences from {fasta_file}")

    embeddings = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i+batch_size]
            
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # Mean pooling of the last hidden state (excluding special tokens)
            # attention_mask: 1 for tokens, 0 for padding
            attention_mask = inputs['attention_mask']
            
            # We want to average over the sequence length, excluding padding
            # hidden_states: [batch, seq_len, hidden_dim]
            last_hidden_state = outputs.last_hidden_state
            
            # Mask padding
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # Sum embeddings
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            
            # Sum masks (clamp to avoid division by zero)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Mean
            batch_embeddings = sum_embeddings / sum_mask
            
            embeddings.append(batch_embeddings.cpu().numpy())

    all_embeddings = np.concatenate(embeddings, axis=0)
    


    base_name = os.path.basename(fasta_file).split('.')[0]
    
    # Save as .dat (fvecs format)
    dat_path = os.path.join(output_dir, f"{base_name}.dat")
    print(f"Saving to {dat_path}...")
    with open(dat_path, "wb") as f:
        dim = all_embeddings.shape[1]
        for vec in all_embeddings:
            f.write(struct.pack('<i', dim))
            f.write(vec.astype('<f4').tobytes())
    print(f"Saved dat file to {dat_path}")
    
    # Save IDs as well for reference
    with open(os.path.join(output_dir, f"{base_name}_ids.txt"), 'w') as f:
        for seq_id in ids:
            f.write(f"{seq_id}\n")

    return all_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein embeddings using ESM-2")
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--output", type=str, default="embeddings_output", help="Output directory")
    parser.add_argument("--model", type=str, default="facebook/esm2_t6_8M_UR50D", help="Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    generate_embeddings(args.model, args.fasta, args.output, args.batch_size)
