# Remote-Homologs: Neural Vector Search for Protein Homology Detection

## Step 1: Execute Embedding Generation (Database Preparation)

```bash
python src/protein_embed.py \
  -i dataset/swissprot_50k.fasta \
  -o embeddings/vectors.dat \
  --model esm2_t6_8M_UR50D
```

**Inputs:**
- `-i` (input FASTA): Protein sequences to embed (SwissProt database)
- `--model`: ESM-2 variant (t6_8M_UR50D = 320 dimensions)

**Outputs:**
- `vectors.dat`: Binary NumPy file (N × 320 matrix, N = # proteins)
- `ids.txt`: Index mapping (row i → Protein ID at position i)

**Purpose**: Generate ESM-2 embeddings for entire database (offline stage, reusable)

## Step 2: Execute Search & Comparative Evaluation

```bash
python src/protein_search.py \
  -d embeddings/vectors.dat \
  -q dataset/targets.fasta \
  -o results/search_results.txt \
  -method all
```

**Inputs:**
- `-d` (database vectors): Embeddings file from Step 1 (vectors.dat)
- `-q` (queries): FASTA file with query proteins (same format as database)
- `-method`: ANN algorithm choice (all|lsh|hypercube|ivfflat|ivfpq)

**Outputs:**
- `search_results.txt`: Results with rankings, scores, and timing metrics
- Timing metrics per query
- Performance comparison across ANN methods

**Three-Stage Search Pipeline**:
1. **Index Loading/Building**: Load or construct ANN index from vectors.dat
2. **Query Embedding**: Compute ESM-2 embeddings for query proteins (on-the-fly)
3. **Search Execution**: Find K nearest neighbors in database, measure QPS (queries/second)

### Four-Stage Embedding Production Pipeline

1. **Tokenization & Truncation** (Amino Acid → Token Conversion)
   - Alphabet: 20 standard amino acids + 3 special characters (start, stop, unknown) = 23 tokens
   - Each amino acid maps to integer 0-22
   - Max sequence length: 1,022 amino acids (transformer window)
   - Sequences longer than 1,022 are truncated (rare in practice)
   - Result: Integer sequence of tokens ready for embedding lookup

2. **Batching** (Length-Based Grouping for GPU Memory)
   - Group proteins by sequence length (shorter sequences together)
   - Benefit: Reduces padding waste (all sequences in batch get padded to max-in-batch length)
   - Example: Batch with lengths [100, 105, 102] padded to 105 vs batch with [100, 500, 1000] padded to 1,000
   - Memory optimization: Prevents large sequences from forcing expensive padding on all batch members
   - Result: Efficient GPU utilization, ~4 GB peak memory for 570K proteins in multiple batches

3. **Inference** (Forward Pass Through ESM-2)
   - Load ESM-2 (t6_8M_UR50D) onto GPU
   - For each token sequence: forward pass through 6 Transformer layers
   - Each layer: attention + feedforward operations
   - Output: Hidden state H ∈ R^(L×320) per protein
   - Timing: ~2-5ms per protein on GPU (batch processing)
   - Total: 570K proteins → ~2-3 hours on NVIDIA GPU

4. **Aggregation** (Mean Pooling to Protein Vectors)
   - Input: Per-amino acid embeddings H ∈ R^(L×320)
   - Operation: v = (1/L) × ∑(i=1 to L) h_i
   - Output: Single vector v ∈ R^320 per protein
   - Interpretation: v encodes protein's structural and functional profile
   - Storage: 570K proteins × 320 dimensions × 4 bytes = 728 MB (float32)


### Critical Implementation Parameters

**Layer 6 (Final Layer)**:
- Contains most abstract semantic information
- Captures domain structure despite sequence erosion
- Best for remote homology detection

**Truncation at 1024**:
- Maximum sequence length due to O(L²) attention complexity
- GPU memory constraint: L=1024 uses 3GB, L=2048 uses 12GB
- Affects only ~0.5% of SwissProt (rarely needed)

**Mean Pooling Aggregation**:
- Formula: v = (1/L) × Σ(i=1 to L) h_i
- Fixed output: always 320 dimensions (regardless of protein length)
- Balanced contribution: all positions contribute equally
- Preserves domain similarity: shared domains → similar averages
- Better than max pooling (keeps all information) or concatenation (scales poorly)

- GPU memory: 3-4 GB peak
- Output: 728 MB (vectors.dat) + 2 MB (ids.txt)

