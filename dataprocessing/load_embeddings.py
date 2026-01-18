import struct
import numpy as np
import os

def read_dat(filename):
    """
    Reads a .dat file (fvecs format: int32 dimension, followed by float32 data).
    Returns a numpy array of shape (N, dim).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    # Read the dimension from the first 4 bytes
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            return np.array([])
        dim = struct.unpack('<i', dim_bytes)[0]
    
    # Calculate file size to determine N
    file_size = os.path.getsize(filename)
    # Each vector is 4 bytes (dim) + 4*dim bytes (floats)
    vec_size = 4 + 4 * dim
    
    if file_size % vec_size != 0:
        print(f"Warning: File size {file_size} is not a multiple of vector size {vec_size}")
    
    n_vecs = file_size // vec_size
    
    # Efficiently read using numpy
    # We read (N, dim+1) int32s first. data[0] is dim.
    
    with open(filename, 'rb') as f:
        # Read the whole file
        data = np.fromfile(f, dtype=np.int32)
    
    n_vecs_read = data.shape[0] // (dim + 1)
    if n_vecs_read != n_vecs:
        print(f"Warning: calculated n_vecs {n_vecs} differs from read elements {n_vecs_read}")
        
    data = data.reshape((n_vecs_read, dim + 1))
    
    # Check dimensions
    if not np.all(data[:, 0] == dim):
        raise ValueError("Inconsistent dimensions in dat file")
        
    # Drop the first column (dimensions) and view as float
    # We must treat the memory of the remaining columns as floats
    vecs_int = data[:, 1:]
    vecs = vecs_int.view(np.float32)
    
    return vecs

def load_ids(filename):
    """Loads a list of IDs from a text file."""
    if not os.path.exists(filename):
        print(f"Warning: ID file {filename} not found.")
        return []
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def load_pfam_map(filename):
    """
    Loads Pfam mappings from a TSV file.
    Expected format: <ProteinID> \t <PfamID> (or similar)
    Returns a dict: ProteinID -> Set of PfamIDs (or single string)
    """
    mapping = {}
    if not os.path.exists(filename):
        return mapping
        
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                prot_id = parts[0]
                pfam_id = parts[1]
                # Assuming one-to-one or one-to-many?
                # Usually proteins can have multiple domains.
                if prot_id not in mapping:
                    mapping[prot_id] = set()
                mapping[prot_id].add(pfam_id)
    return mapping
