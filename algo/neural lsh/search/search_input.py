"""
Search I/O Module
=================
Functions for loading data and saving results.
"""

import os
import pickle
import numpy as np
from typing import Dict, Tuple


def load_index_and_model(index_path: str) -> Tuple[Dict, 'NeuralLSH_MLP', 'torch.device']:
    """
    Φόρτωση του ευρετηρίου και του εκπαιδευμένου μοντέλου.
    
    Args:
        index_path: Διαδρομή προς το αρχείο του ευρετηρίου
        
    Returns:
        Tuple[index_data, model, device]
    """
    import torch
    from build.index_model.mlp_model import NeuralLSH_MLP
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Το αρχείο ευρετηρίου δεν βρέθηκε: {index_path}")
    
    print(f"Φόρτωση ευρετηρίου από: {index_path}")
    
    with open(index_path, 'rb') as f:
        index_data = pickle.load(f)
    
    # Αναδημιουργία του μοντέλου
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = index_data['model_config']
    model = NeuralLSH_MLP(
        input_dim=model_config['input_dim'],
        num_layers=model_config['num_layers'],
        nodes_per_layer=model_config['nodes_per_layer'],
        output_dim=model_config['output_dim']
    )
    
    # Φόρτωση των βαρών του μοντέλου
    model.load_state_dict(index_data['model_state_dict'])
    model = model.to(device)
    model.eval()  # Θέσιμο του μοντέλου σε κατάσταση αξιολόγησης
    
    print(f"Μοντέλο φορτώθηκε (device: {device})")
    print(f"  Input dim: {model_config['input_dim']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Buckets: {model_config['output_dim']}")
    
    return index_data, model, device


def load_data(data_path: str) -> np.ndarray:
    """
    Φόρτωση του συνόλου δεδομένων (απαραίτητο για υπολογισμό πραγματικών αποστάσεων).
    
    Args:
        data_path: Διαδρομή προς το αρχείο δεδομένων
        
    Returns:
        numpy array με τα δεδομένα (n_samples, dim)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Το αρχείο δεδομένων δεν βρέθηκε: {data_path}")
    
    print(f"Φόρτωση δεδομένων από: {data_path}")
    
    if data_path.endswith('.npy'):
        data = np.load(data_path)
    elif data_path.endswith('.idx3-ubyte'):
        # Support MNIST format
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataparser.mnist_parser import MNISTParser
        images = MNISTParser.read_images(data_path)
        data = MNISTParser.flatten_images(images)
    elif data_path.endswith('.fvecs'):
        # Support SIFT format
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataparser.sift_parser import SIFTParser
        data = SIFTParser.read_fvecs(data_path)
    else:
        raise NotImplementedError(f"Μορφή αρχείου {data_path} δεν υποστηρίζεται. "
                                  "Χρησιμοποιήστε .npy, .idx3-ubyte, ή .fvecs αρχεία.")
    
    print(f"Φορτώθηκαν {len(data)} δεδομένα με διάσταση {data.shape[1]}")
    return data


def load_queries(queries_path: str, max_queries: int = None) -> np.ndarray:
    """
    Φόρτωση των ερωτημάτων (queries).
    
    Args:
        queries_path: Διαδρομή προς το αρχείο με τα queries
        max_queries: Optional maximum number of queries to load (for matching cache)
        
    Returns:
        numpy array με τα queries (n_queries, dim)
    """
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Το αρχείο queries δεν βρέθηκε: {queries_path}")
    
    print(f"Φόρτωση queries από: {queries_path}")
    
    if queries_path.endswith('.npy'):
        queries = np.load(queries_path)
    elif queries_path.endswith('.idx3-ubyte'):
        # Support MNIST format
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataparser.mnist_parser import MNISTParser
        images = MNISTParser.read_images(queries_path)
        queries = MNISTParser.flatten_images(images)
    elif queries_path.endswith('.fvecs'):
        # Support SIFT format
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataparser.sift_parser import SIFTParser
        queries = SIFTParser.read_fvecs(queries_path)
    else:
        raise NotImplementedError(f"Μορφή αρχείου {queries_path} δεν υποστηρίζεται. "
                                  "Χρησιμοποιήστε .npy, .idx3-ubyte, ή .fvecs αρχεία.")
    
    # Limit to max_queries if specified
    if max_queries is not None and len(queries) > max_queries:
        queries = queries[:max_queries]
        print(f"Φορτώθηκαν {len(queries)} queries (limited to max_queries={max_queries}) με διάσταση {queries.shape[1]}")
    else:
        print(f"Φορτώθηκαν {len(queries)} queries με διάσταση {queries.shape[1]}")
    return queries