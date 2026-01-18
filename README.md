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

## Step 3: Execute BLAST (Ground Truth)

```bash
# Create BLAST database
makeblastdb -in dataset/swissprot.fasta \
  -dbtype prot \
  -out database/swissprot_db

# Execute search
blastp -db database/swissprot_db \
  -query dataset/targets.fasta \
  -outfmt 6 \
  -out results/blast_results.tsv
```

**Database Creation**:
- Input: Protein sequences in FASTA format
- Output: BLAST-formatted database files (swissprot_db.*)
- Flag `-dbtype prot`: Protein database (not nucleotide)

**Search Execution**:
- Input: Query proteins (FASTA format)
- Output: Tabular results (TSV, columns: query, hit, identity%, alignment_length, mismatches, gap_opens, E-value, bit_score)
- Flag `-outfmt 6`: Tabular output without headers (ideal for automated parsing)

**Purpose**:
- BLAST provides ground truth homology reference
- Enables Recall@N evaluation: compares ANN results against BLAST hits
- Filtering criteria: E-value < 0.001, identity < 30% (twilight zone focus)

### Step 3.1: Parse & Filter BLAST Results

```bash
python search_benchmark/parse_blast_results.py \
  -i results/blast_results.tsv \
  -o results/blast_ground_truth.pkl \
  -evalue 0.001 \
  -top_n 100
```

**Processing Pipeline:**
1. **Read** blast_results.tsv (tab-separated columns, no headers)
2. **Filter** by E-value < 0.001 (remove statistical noise)
3. **Sort** hits by Bit-Score (descending, higher = better alignment)
4. **Select** Top-N results per query (e.g., top 100)
5. **Store** as S_BLAST reference set (pickle format: {query_id: [hit_id1, hit_id2, ...]})

**Output Usage:**
- Ground truth reference set for Recall@N evaluation
- Enables comparison: |ANN_hits ∩ BLAST_hits| / |BLAST_hits|

## Step 4: Compute Recall@N & QPS Metrics

```bash
python search_benchmark/compute_metrics.py \
  -ann_results results/search_results.txt \
  -blast_ground_truth results/blast_ground_truth.pkl \
  -o results/metrics_summary.txt
```

**Metrics Computed:**

1. **Recall@N** (Accuracy - % of ground truth recovered)
   - Formula: Average Recall@N = (1/|Q|) × Σ(q∈Q) |ANN_q ∩ BLAST_q| / N
   - Per-query: Recall@N = |ANN_q ∩ BLAST_q| / |BLAST_q|
   - Interpretation: % of BLAST hits found in top-N ANN results
   - Range: 0.0 (no recovery) to 1.0 (perfect recovery)

2. **QPS** (Throughput - queries per second)
   - Formula: QPS = Total Queries / Total Search Time (seconds)
   - Measures: System throughput capability
   - Interpretation: Higher QPS = faster search

**Outputs:**
- `metrics_summary.txt`: Per-method performance table
  - Columns: Method, Recall@100, QPS, Index Time (s), Memory (MB)

## Step 5: Hyperparameter Tuning (Grid Search)

```bash
python search_benchmark/hyperparameter_grid_search.py \
  -method all \
  -vectors embeddings/vectors.dat \
  -queries dataset/targets.fasta \
  -blast_ground_truth results/blast_ground_truth.pkl \
  -o results/hyperparameter_tuning.csv
```

**Grid Search Parameters:**

1. **LSH**: hash_bits=[8, 12, 16, 20], num_tables=[10, 20, 30]
2. **Hypercube**: radius=[0.5, 1.0, 1.5, 2.0], partitions=[16, 32, 64]
3. **IVF-PQ**: nlist=[16, 32, 64, 128], nprobe=[1, 2, 4, 8]

**Output:**
- Recall vs QPS trade-off curves per method
- Optimal parameter configuration (best Recall for acceptable latency)
- CSV with all tested combinations and their metrics

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

### Implementation: PyTorch + ESM Library

**Basic Code Example**:
```python
import torch, esm
from Bio import SeqIO
import numpy as np

# 1. Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()

# 2. Prepare sequences (with truncation)
sequences = []
for record in SeqIO.parse("input.fasta", "fasta"):
    seq = str(record.seq).upper()[:1022]  # Truncate at 1022
    sequences.append((record.id, seq))

# 3. Process in batches
all_embeddings = []
for batch in [sequences[i:i+64] for i in range(0, len(sequences), 64)]:
    labels, strs, tokens = batch_converter(batch)
    with torch.no_grad():
        results = model(tokens, repr_layers=[6])
    
    # 4. Mean pooling
    embeddings = results["representations"][6].mean(dim=1)
    all_embeddings.append(embeddings.cpu().numpy())

# Save outputs
np.save("vectors.dat", np.vstack(all_embeddings))
with open("ids.txt", "w") as f:
    f.write("\n".join([seq[0] for batch in [sequences[i:i+64] for i in range(0, len(sequences), 64)] for seq in batch]))
```

**Performance (A100 GPU, 570K proteins)**:
- Throughput: 64-128 proteins/second
- Total time: ~2-3 hours

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

## Overview
This project addresses a critical problem in computational biology: **detecting remote homologs when classical sequence identity fails**.

### What is Homology?
- **Homology (ομολογία)**: A binary relationship - proteins either ARE or ARE NOT homologous
- Indicates **shared evolutionary ancestor**
- Enables **knowledge transfer**: finding a homolog lets us annotate a new protein
- Usually inferred from: sequence similarity, 3D structure, biological function

### The Problem: The Twilight Zone
- **Sequence Explosion**: NGS and metagenomics generate hundreds of millions of protein sequences (UniProtKB/TrEMBL)
- **Structure Bottleneck**: Experimental structure determination (X-ray, Cryo-EM, NMR) is slow → limited solved structures in PDB
- **Annotation Gap**: Most sequences lack functional annotation (vs. curated SwissProt)
- **Classical Homology Failure**: Sequence identity-based methods fail below ~20% identity (the "twilight zone")

### Remote Homologs: The Missing Homologous Proteins
- **Remote homologs**: Homologous proteins with alignment identity 20-30%
- **The Paradox**: They ARE truly homologous (share evolutionary ancestor, similar 3D structure, related function)
- **Hidden by Evolution**: Despite being homologous, sequence erosion makes them invisible to alignment methods
- **Domain Persistence**: They share structural domains (self-contained regions that fold independently)
- **The Opportunity**: If we can detect shared domains instead of relying on sequence alignment, we find TRUE homologs

### Why BLAST Fails for Remote Homologs (Technical Details)
BLAST is a heuristic algorithm (fast approximation of Smith-Waterman) that works via:
1. **Seeding**: Breaks query sequence into k-mers (k=3 for proteins), finds database k-mers scoring above threshold
2. **Extension**: Extends matches bidirectionally while alignment score increases; stops when score drops (drop-off threshold)
3. **Scoring**: Computes HSP alignment score (S) - algebraic sum of BLOSUM substitution scores minus gap penalties
4. **Statistical Filtering**: Calculates E-value (expected random matches with score ≥ S in database)
5. **Result**: HSPs (High-Scoring Segment Pairs) filtered by E-value < 0.01 (significant), ranked by score

**The Two-Tier Problem in Twilight Zone**:
- **Tier 1 - Seeding Bottleneck** (k-mer matching fails first):
  * K-mers require high sequence similarity to exceed threshold
  * In twilight zone (20-30% identity), k-mers don't match well
  * Result: No seeds → no extension → no HSP computation
  * Consequence: S and E-value never calculated for remote homologs
- **Tier 2 - Statistical Filtering** (would work IF seeding succeeded):
  * E-value < 0.01 correctly identifies significant alignments
  * But this filtering never reached for remote homologs (seeding failed first)
  * Even low E-values can't rescue algorithms with no input HSPs

**Our Advantage - Vector-based Search**: No k-mer bottleneck
- Embeddings capture domain structure beyond character sequences
- ANN search in vector space finds similar proteins even with severe erosion
- Similarity ranking analogous to E-value filtering (both measure "significance")
- Domains remain recognizable even when twilight zone erosion is severe
- Critical insight: Remote homologs have similar embeddings BECAUSE shared domain structure survived erosion

### Our Solution: Neural Vector Search for Homology
Instead of relying solely on sequence identity and k-mer seeding, we:
1. **Generate neural embeddings** using ESM-2 (Evolutionary Scale Modeling)
   - Captures domain structure and semantic similarity
   - Preserves functional relationships despite sequence erosion
   - Works with embedding vectors, not character k-mers
   - **HOW ESM-2 WORKS: Transformer Attention Mechanism**
     * Parallel processing: Each amino acid "converses" with ALL others simultaneously
     * Not linear (left-to-right) but parallel attention across entire sequence
     * Three learned vectors per amino acid position:
       - **Query (Q)**: "What characteristics am I seeking in other amino acids?"
       - **Key (K)**: "What characteristics do I broadcast so others recognize me?"
       - **Value (V)**: "What information do I convey when others attend to me?"
     * Attention computation: Attention(Q,K,V) = Softmax(Q·K^T/√d)V
       - When Q matches K strongly → high attention weight → strong interaction
       - Captures which amino acids should "look at" each other
       - Example: Disulfide bond: C's Query matches C's Key → they attend to each other
     * Result: Learned attention patterns encode domain structure
   - **BATCH 13: STATIC vs CONTEXTUAL REPRESENTATION**
     * Static representation (traditional lookup tables):
       - Alanine always has the same vector, regardless of context
       - Cannot distinguish if Alanine is in hydrophobic core vs surface
       - Cannot recognize different structural/functional roles
     * Contextual representation (Transformer):
       - Each amino acid position generates DIFFERENT vector depending on context
       - Output vector encodes: identity + environment + role
       - For Alanine: different if buried vs exposed, different if structural vs flexible
     * Multi-dimensional encoding in ESM-2:
       - Physicochemical properties: charge, size, polarity
       - Structural role: is it in helix? turn? loop? domain boundary?
       - Functional role: does it bind metal? is it catalytic? part of active site?
     * ESM-2 Model Specifications (t6_8M_UR50D):
       - Architecture: 6 Transformer layers (lightest variant for efficiency)
       - Parameters: ~8 million (small enough for GPU inference)
       - Embedding dimension (d): 320 (output vector size per amino acid)
       - Training data: UniRef50 (50% identity clustering to ensure diversity)
       - Output: For protein length L, matrix H ∈ R^(L×320)
   - **BATCH 14: MEAN POOLING - FROM PER-AMINO ACID TO PROTEIN-LEVEL EMBEDDINGS**
     * Problem: Each amino acid has 320-dim embedding, but we need ONE vector per protein for ANN search
     * Solution: Mean pooling (averaging over sequence length)
       - For protein of length L: H ∈ R^(L×320)
       - Compute: v = (1/L) Σ h_i, where h_i is row i of H
       - Result: v ∈ R^320 (single vector per protein)
     * Why mean pooling preserves similarity:
       - If two proteins have similar domain structures → similar amino acid embeddings
       - Similar embeddings → similar mean → ||v1 - v2||_2 small (Euclidean distance)
       - Even if sequences diverge (remote homologs), domain structure similarity → embedding similarity
     * ESM-2 trained on convergent evolution:
       - Exposed to many protein family variants with same structure/function
       - Exposed to convergent evolution examples: different sequences, identical structure/function
       - Result: Model learns structure/function is MORE conserved than sequence
     * Outcome:
       - Structurally/functionally related proteins → small ||v1 - v2||_2
       - Remote homologs (same domains, different sequences) → still clustered
       - Vector v serves as input to ANN search algorithms (LSH, IVF-PQ, Neural-LSH)
   - **LINGUISTIC ANALOGY**: Context-dependent embeddings
     * Static methods (BLAST k-mers): Always same vector, ignores context
     * ESM-2 embeddings: Different vector for same amino acid depending on context
     * Example: Cysteine (C) encodes differently depending on:
       - Structural context: disulfide bond role (C···C nearby in 3D)
       - Catalytic context: catalytic triad role (C···H···D coordination)
     * Result: Embeddings encode ROLE not just IDENTITY
     * This is why remote homologs cluster despite low sequence identity
   - **BATCH 12: HOW ESM-2 LEARNS STRUCTURE - Masked Language Modeling (MLM)**
     * Training objective: Mask 15% of amino acids, predict the hidden residues
     * Why this teaches structure:
       - To predict Tryptophan (W) at position i: model learns environment is hydrophobic
       - To predict Proline (P) at position i: model learns backbone must be rigid
       - To predict Cysteine (C) at position i: model learns to find disulfide bond partners (long-range!)
     * Emergent structure discovery: Attention maps perfectly correlate with Contact Maps (3D residue proximity)
     * Information compression principle: Transformer compressed 1D sequence into 3D structure
       - Structure is the OPTIMAL representation for solving the masking prediction problem
       - Model never saw 3D coordinates, yet discovered protein folding spontaneously
     * Consequence: Domain structure persists across evolutionary distance (20-30% identity)
       - Homologous proteins maintain similar attention patterns → similar embeddings
       - ANN search finds them in embedded space
2. **BATCH 15: DISTANCE METRICS AND ANN NECESSITY**
   - High dimensionality of embeddings: d = 320 (ESM-2 t6_8M) or 1280 (larger variants)
   - Distance metrics for similarity:
     * Euclidean (L2): dist(v,u) = ||v - u||₂ (standard distance)
     * Cosine similarity (preferred): measures angle between vectors
       - Why cosine: Vector length often has no biological meaning
       - Angle captures structure/function similarity better than magnitude
       - Normalized vectors: focus on direction not scale
   - **Why ANN is necessary**:
     * SwissProt database: ~570,000 proteins
     * Each protein: 320-dim embedding (or higher)
     * Search space: 570,000 × 320 = 182 million dimensions
     * Exhaustive search: Too slow for real-time queries
     * Brute force: k nearest neighbors = O(N×d) per query
     * Solution: Approximate Nearest Neighbor algorithms for sub-linear speed
   - **Algorithms compared**:
     * LSH (Locality-Sensitive Hashing): Fast, data-agnostic randomized projections
     * Hypercube: Data-dependent partitioning, adapts to data structure
     * IVF (Inverted File): Coarse partitioning + refinement, balanced speed/accuracy
     * Neural-LSH: Learned quantization scheme optimized for learned embeddings
     * Goal: Comparative evaluation on biological data (remote homolog detection)
   
   **BATCH 16: EVALUATION FRAMEWORK - S_BLAST vs S_ANN**
   - **Reference Standard (S_BLAST)**: Top-N results from BLAST search
     * Run BLAST on each query protein against SwissProt
     * Select Top-N results by BLAST score
     * These become ground truth for evaluation
   - **ANN Prediction (S_ANN)**: Top-N nearest neighbors in embedding space
     * Run ANN search on same query protein
     * Find Top-N nearest proteins by distance/similarity
     * Same N as BLAST for fair comparison
   - **Success Metric**: How well do S_ANN and S_BLAST align?
     * Overlap = true positives (detected by both methods)
     * BLAST-only = false negatives for ANN
     * ANN-only = false positives for ANN
     * Goal: S_ANN should find proteins BLAST finds (especially remote homologs)
   
   **BATCH 17: RECALL@N METRIC - PRIMARY EVALUATION**
   - **Definition**: For query with |S_BLAST| = N true neighbors:
     * **Recall@N = |S_ANN ∩ S_BLAST| / |S_BLAST|**
   - **Interpretation**:
     * Measures percentage of true homologs (per BLAST) that ANN retrieves
     * Ignores ranking order; only counts set membership
     * Set intersection: both methods must agree protein is homolog
   - **Value Range**: [0, 1]
     * 1.0 = perfect recall (found all BLAST homologs)
     * 0.5 = found half of BLAST homologs (missed half)
     * 0.0 = found no BLAST homologs (complete failure)
   - **Biological Meaning**:
     * High Recall@N = ANN captures homologs BLAST captures
     * Essential for twilight zone: can ANN find what BLAST finds?
     * Not ranking-dependent: order doesn't matter, only retrieval

   **BATCH 18: REMOTE HOMOLOGY CANDIDATE DEFINITION**
   - **Definition**: A protein pair is a remote homology candidate when it satisfies **all three**:
     1. **Low sequence identity** (Identity < 30%)
        * Within twilight zone, impossible for k-mer seeding to detect
        * Yet proven to be evolutionary homolog
     2. **Proximity in embedding space** (small distance, high cosine similarity)
        * ESM-2 embeddings place similar structures close
        * Domain organization survives sequence divergence
     3. **Biological evidence** from UniProt annotations:
        * Common Pfam domain (shared functional module)
        * Matching EC number (same enzymatic function)
        * Similar GO terms (same biological process/component)
   - **Multi-Modal Validation**:
     * Geometry (embeddings) + Function (annotations) = true remote homolog
     * Geometry alone = potentially false positive
     * Function evidence = distinguishes true homologs from coincidental similarity
   **BATCH 19: ERROR ANALYSIS - FALSE POSITIVES vs TRUE POSITIVES**
   - **False Positives** (embedding false alarms):
     * Characteristics: Small embedding distance + low BLAST identity + NO functional annotation
     * Symptoms: ANN predicts homology but Pfam/EC/GO disagree
     * Likely causes:
       - ESM-2 limitations: Cannot model specific protein families equally well
       - Data quality: Incomplete or erroneous UniProt annotations
       - Distance metric: Cosine similarity may not capture true biological distance
     * Impact: Lowers precision (false claims of homology)
   - **True Positives** (embedding victories):
     * BLAST fails: Identity < 30%, E-value insignificant → sequence method defeated
     * ANN succeeds: Embedding proximity clusters protein correctly
     * Annotation confirms: Shared Pfam domain OR matching EC number OR similar GO
     * Meaning: Embeddings revealed *hidden* homology (what BLAST missed)
   - **Diagnostic Insight**:
     * False Positives show: Which protein families ESM-2 struggles with
     * True Positives prove: Learned embeddings overcome sequence alignment limitations
     * Together: Illuminate where method excels and where it fails

   **BATCH 20: COMPLETE EXECUTION PIPELINE**
   - **Stage 1: Database Preparation**
     * Input: swissprot.fasta (~570,000 protein sequences)
     * Generate ESM-2 embeddings for entire database
     * Output: 570,000 × 320 embedding matrix
     * Storage: Serialized embeddings for fast loading
   - **Stage 2: Index Construction**
     * Build four ANN indices on complete embedding matrix:
       - LSH: Fast randomized projections
       - Hypercube: Data-dependent partitioning
       - IVF: Coarse clustering + fine refinement
       - Neural-LSH: Learned quantization for embeddings
   - **Stage 3: Query Search** (both methods, same queries)
     * For each query protein:
       1. Execute ANN search: retrieve Top-N neighbors
       2. Execute BLAST search: retrieve Top-N results
     * Collect S_ANN and S_BLAST result sets
   - **Stage 4: Performance Metrics**
     * **Recall@N = |S_ANN ∩ S_BLAST| / |S_BLAST|**
       - Retrieval success: what fraction of BLAST results found by ANN?
     * **QPS (Queries Per Second)**
       - Speed metric: throughput and latency comparison
       - ANN speed advantage over exhaustive BLAST search
   - **Stage 5: Biological Evaluation Report**
     * Analyze true positives: BLAST missed, ANN found, annotations validate
     * Analyze false positives: ANN found but no biological evidence
     * Case studies: selected twilight zone successes and failures
     * Quantify method value and limitations

3. **Search with Approximate Nearest Neighbors** (LSH, IVF-PQ, Neural-LSH)
   - Efficiently find similar proteins in embedding space
   - Detects shared domains missed by alignment
   - No k-mer seeding bottleneck
   - Ranking by structural similarity (not k-mer discrimination)
4. **Compare with BLAST** (sequence identity baseline)
   - Validate when neural methods discover remote homologs
   - Understand where each method excels/fails
5. **Validate with UniProt/SwissProt Annotations**
   - EC numbers, GO terms, Pfam domains distinguish TRUE from FALSE positives
   - Annotation consistency metrics confirm biological relevance

### Validation & Evaluation
Each homolog prediction is validated using UniProt/SwissProt metadata:
- **Functional annotations**: Does query protein have relevant function annotation?
- **EC numbers**: Do both proteins catalyze same enzyme reaction class?
- **GO terms**: Do they share biological process, molecular function, cellular component?
- **Pfam/InterPro domains**: Do they share structural/functional domains?
- **Cross-references**: PDB structures, STRING interactions, literature support

**TRUE POSITIVES vs FALSE POSITIVES:**
- TRUE: Found homologs share EC/GO/Pfam annotations (prove true homology)
- FALSE: Embedding similarity but no functional/structural overlap (noise)

### Understanding BLAST Result Metrics & Identity Thresholds
When BLAST reports alignments, each result includes alignment statistics:

**Identity Thresholds for Homology Classification:**
For sufficiently long alignments, identity percentage determines homology confidence:

| Identity Range | Classification | Status | Confidence |
|---|---|---|---|
| **≥ 30%** | **Clear Homology** | Easily detected | High confidence (few false positives) |
| **20-30%** | **TWILIGHT ZONE** | Remote homologs (truly homologous) | Medium confidence (harder to validate) |
| **< 20%** | **Indistinguishable** | Random similarity | No confidence (cannot separate signal from noise) |

**Identities (Ταυτότητα)**: Number of positions with identical amino acids
- Example: 58/136 (43%) = 43% sequence identity
- Critical for twilight zone: 20-30% identity = remote homologs that ARE truly homologous
- Direct measure of sequence conservation at aligned positions
- Direct measure of sequence conservation despite evolutionary erosion

**Positives (Ομοιότητα)**: Identities + conservative substitutions (positive BLOSUM scores)
- Example: 85/136 (62%) = 62% similarity (includes physicochemical matches)
- Shows alignment quality: even low-identity alignments may have structure
- Ratio of Positives/Identities reveals if alignment is real or random

**Gaps (Κενά)**: Insertions/deletions (indels) in the alignment
- Example: 2/136 (1%) = only 1% gaps
- Low gaps → likely true homology (single structural domain)
- High gaps → possible domain rearrangement or alignment artifact

**Why This Matters for Remote Homologs:**
- Twilight zone homologs have 20-30% identity BUT high positives and low gaps
- This pattern indicates: low sequence identity BUT structural conservation
- **NEURAL EMBEDDINGS DETECT THIS:** they capture the structure hidden in low identity
- BLAST seeding fails before computing these metrics, so never evaluates them

### Key Concepts
- **Homology**: Binary property - proteins ARE or ARE NOT homologous (from common ancestor)
- **Domains**: Self-contained structural regions that fold independently; persist despite sequence erosion
- **Twilight Zone** (20-30% identity): Remote homologs with low sequence similarity but TRUE homology (shared domains)
- **Evolutionary Erosion**: Sequence similarity erodes faster than structure/domains; alignment methods fail but domains remain similar
- **Neural Embeddings**: Capture domain structure beyond raw sequence, enabling remote homolog detection
- **Annotation-driven validation**: EC/GO/Pfam confirm biological relevance and distinguish true homologs
- **Complementary to AlphaFold**: While AlphaFold predicts 3D structure, we transfer functional knowledge via shared domains