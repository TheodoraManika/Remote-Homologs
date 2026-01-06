#!/usr/bin/env python3
"""
BLAST Wrapper for Protein Homolog Search
Interfaces with NCBI blastp executable to find remote homologs in SwissProt database

Key BLAST Output Columns Used:
================================
1. evalue (Expect Value)
   - Filter: Remove E-value > 0.01 (noise threshold)
   - Statistically significant hits only
   
2. bitscore (Alignment Quality Score)
   - Sort: Descending order (higher = better)
   - Provides Top-N ranking by alignment quality
   
3. pident (Percent Identity)
   - Categorize results:
     * >= 30%: High-confidence homologs
     * 20-30%: Twilight Zone (remote homologs, BLAST weak signal)
     * < 20%: Below detection threshold (BLAST typically fails)

Processing Pipeline:
====================
Step 1: Load blast_results.tsv (from -outfmt 6)
Step 2: Filter by evalue (> 0.01 removed)
Step 3: Sort by bitscore (descending)
Step 4: Select Top-N → Ground Truth set SBLAST
Step 5: Classify by pident for analysis
"""

import subprocess
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import csv


@dataclass
class BlastHit:
    """Represents a single BLAST alignment hit"""
    query_id: str
    subject_id: str
    percent_identity: float       # Percent identity (Identities/alignment_length * 100)
    alignment_length: int         # Length of alignment
    mismatches: int               # Number of mismatches
    gap_opens: int                # Number of gap openings
    query_start: int              # Start position in query
    query_end: int                # End position in query
    subject_start: int            # Start position in subject
    subject_end: int              # End position in subject
    evalue: float                 # E-value (Expect Value): statistical significance
                                  # Lower values = more reliable results
    bitscore: float               # Bit score: derived from raw score S
                                  # Higher values = better alignment quality
    # Extended BLAST metrics for detailed analysis
    identities: int = 0           # Number of positions with identical amino acids
    positives: int = 0            # Identities + positively scored BLOSUM substitutions
    gaps: int = 0                 # Number of gaps/indels in alignment
    
    def get_identity_count(self) -> int:
        """Get absolute count of identical positions"""
        return self.identities if self.identities > 0 else int(self.percent_identity * self.alignment_length / 100)
    
    def get_positives_count(self) -> int:
        """Get absolute count of positive matches"""
        return self.positives if self.positives > 0 else self.identities
    
    def classification(self) -> str:
        """
        Classify hit based on Twilight Zone thresholds
        
        Classification:
        - identity >= 30%: Clear homology signal
        - 20% <= identity < 30%: Twilight Zone (remote homologs, BLAST weak signal)
        - identity < 20%: Below detection threshold (BLAST typically fails)
        
        Returns:
            One of: "high_confidence", "twilight_zone", "non_homologous"
        """
        if self.percent_identity >= 30.0:
            return "high_confidence"
        elif 20.0 <= self.percent_identity < 30.0:
            return "twilight_zone"
        else:
            return "non_homologous"
    
    def is_twilight_zone(self) -> bool:
        """Check if this hit is in the Twilight Zone (20-30% identity)"""
        return 20.0 <= self.percent_identity < 30.0
    
    def is_high_confidence(self) -> bool:
        """Check if this hit has clear homology (>= 30% identity)"""
        return self.percent_identity >= 30.0
    
    def is_below_detection(self) -> bool:
        """Check if this hit is below BLAST detection threshold (< 20% identity)"""
        return self.percent_identity < 20.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlastResult:
    """Stores all hits for a single query sequence"""
    query_id: str
    hits: List[BlastHit]
    num_hits: int
    
    def __post_init__(self):
        self.num_hits = len(self.hits)


class Blast:
    """
    BLAST Algorithm Wrapper
    Interfaces with NCBI blastp for protein similarity search
    """
    
    # BLAST output format: tab-separated with specific columns
    BLAST_FORMAT = (
        "6 qseqid sseqid pident length mismatch gapopen "
        "qstart qend sstart send evalue bitscore"
    )
    
    BLAST_COLUMNS = [
        'query_id', 'subject_id', 'percent_identity', 'alignment_length',
        'mismatches', 'gap_opens', 'query_start', 'query_end',
        'subject_start', 'subject_end', 'evalue', 'bitscore'
    ]
    
    def __init__(self,
                 blastp_path: str,
                 database_path: str,
                 queries_path: str,
                 output_path: str = "./blast_output.tsv"):
        """
        Initialize BLAST wrapper
        
        Args:
            blastp_path: Path to blastp executable
            database_path: Path to formatted BLAST database (swissprot)
            queries_path: Path to queries FASTA file
            output_path: Path to output file
        """
        self.blastp_path = blastp_path
        self.database_path = database_path
        self.queries_path = queries_path
        self.output_path = output_path
        
        # Default parameters
        self.evalue_threshold = 1e-5
        self.max_target_seqs = 250
        self.num_threads = 4
        
        # Results storage
        self.results: Dict[str, BlastResult] = {}
        
        # Timing measurement
        self.execution_time = None  # Total BLAST execution time in seconds
    
    def set_parameters(self,
                       evalue: float = 1e-5,
                       max_targets: int = 250,
                       threads: int = 4) -> None:
        """
        Set BLAST search parameters
        
        Args:
            evalue: E-value threshold for reporting hits
            max_targets: Maximum number of target sequences to report
            threads: Number of threads to use
        """
        self.evalue_threshold = evalue
        self.max_target_seqs = max_targets
        self.num_threads = threads
        print(f"BLAST parameters set:")
        print(f"  E-value threshold: {evalue}")
        print(f"  Max target sequences: {max_targets}")
        print(f"  Threads: {threads}")
    
    @staticmethod
    def create_database(fasta_path: str,
                       output_name: str,
                       makeblastdb_path: str = "makeblastdb") -> bool:
        """
        Create BLAST database from FASTA file
        
        Step 1 of BLAST Ground Truth pipeline:
        makeblastdb -in swissprot.fasta -dbtype prot -out swissprot_db
        
        Args:
            fasta_path: Path to input FASTA file (e.g., swissprot.fasta)
            output_name: Output database name (e.g., swissprot_db)
            makeblastdb_path: Path to makeblastdb executable
        
        Returns:
            True if database creation successful
        """
        print(f"Creating BLAST database from {fasta_path}...")
        
        if not os.path.exists(fasta_path):
            print(f"Error: FASTA file not found at {fasta_path}")
            return False
        
        # Build makeblastdb command
        cmd = [
            makeblastdb_path,
            "-in", fasta_path,
            "-dbtype", "prot",
            "-out", output_name
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"makeblastdb failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return False
            
            print(f"Database created successfully: {output_name}")
            return True
            
        except subprocess.TimeoutExpired:
            print("Error: Database creation timed out")
            return False
        except Exception as e:
            print(f"Error creating database: {e}")
            return False
    
    def run_search(self) -> bool:
        """
        Execute BLAST search by calling blastp executable
        
        Step 2 of BLAST Ground Truth pipeline:
        blastp -db swissprot_db -query targets.fasta \
               -outfmt 6 -out blast_results.tsv
        
        Output format: -outfmt 6 (tabular, no headers)
        Columns: qseqid sseqid pident length mismatch gapopen 
                 qstart qend sstart send evalue bitscore
        
        TIMING: Measures total execution time for performance evaluation
        
        Returns:
            True if search completed successfully
        """
        # Validate input files
        if not os.path.exists(self.blastp_path):
            print(f"Error: blastp executable not found at {self.blastp_path}")
            return False
        
        if not os.path.exists(self.database_path):
            print(f"Error: Database not found at {self.database_path}")
            print(f"  Hint: Run create_database() first to generate BLAST database")
            return False
        
        if not os.path.exists(self.queries_path):
            print(f"Error: Queries file not found at {self.queries_path}")
            return False
        
        # Build blastp command with -outfmt 6 (tabular output)
        cmd = [
            self.blastp_path,
            "-query", self.queries_path,
            "-db", self.database_path,
            "-out", self.output_path,
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
            "-evalue", str(self.evalue_threshold),
            "-max_target_seqs", str(self.max_target_seqs),
            "-num_threads", str(self.num_threads),
            "-num_alignments", str(self.max_target_seqs)
        ]
        
        print(f"Running BLAST search...")
        print(f"  Query file: {self.queries_path}")
        print(f"  Database: {self.database_path}")
        print(f"  Output: {self.output_path}")
        print(f"  Format: -outfmt 6 (tabular, no headers)")
        
        try:
            # Measure execution time
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            end_time = time.time()
            
            self.execution_time = end_time - start_time
            
            if result.returncode != 0:
                print(f"BLAST execution failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return False
            
            print(f"BLAST search completed successfully")
            print(f"  Execution time: {self.execution_time:.2f}s")
            print(f"  Output written to: {self.output_path}")
            
            # Parse results
            return self.parse_output()
            
        except subprocess.TimeoutExpired:
            print("Error: BLAST search timed out")
            return False
        except Exception as e:
            print(f"Error running BLAST: {e}")
            return False
    
    def parse_output(self) -> bool:
        """
        Parse BLAST output in TSV format
        
        BLAST Evaluation Pipeline:
        [Step 1] Scoring: blastp calculates alignment score S (from BLOSUM matrix)
        [Step 2] E-value: blastp calculates expect value for statistical significance
        [Step 3] Filtering & Ranking: results filtered and sorted by E-value, then bit score
        
        Returns:
            True if parsing successful
        """
        if not os.path.exists(self.output_path):
            print(f"Error: Output file not found at {self.output_path}")
            return False
        
        print(f"Parsing BLAST output from {self.output_path}...")
        
        self.results.clear()
        
        try:
            with open(self.output_path, 'r') as f:
                reader = csv.DictReader(
                    f,
                    fieldnames=self.BLAST_COLUMNS,
                    delimiter='\t'
                )
                
                for row in reader:
                    # Convert numeric fields
                    hit = BlastHit(
                        query_id=row['query_id'],
                        subject_id=row['subject_id'],
                        percent_identity=float(row['percent_identity']),
                        alignment_length=int(row['alignment_length']),
                        mismatches=int(row['mismatches']),
                        gap_opens=int(row['gap_opens']),
                        query_start=int(row['query_start']),
                        query_end=int(row['query_end']),
                        subject_start=int(row['subject_start']),
                        subject_end=int(row['subject_end']),
                        evalue=float(row['evalue']),
                        bitscore=float(row['bitscore'])
                    )
                    
                    # Add to results
                    if hit.query_id not in self.results:
                        self.results[hit.query_id] = BlastResult(
                            query_id=hit.query_id,
                            hits=[]
                        )
                    
                    self.results[hit.query_id].hits.append(hit)
            
            print(f"Parsed {len(self.results)} queries")
            total_hits = sum(len(r.hits) for r in self.results.values())
            print(f"Total hits: {total_hits}")
            
            return True
            
        except Exception as e:
            print(f"Error parsing BLAST output: {e}")
            return False
    
    def get_results(self, query_id: str) -> Optional[BlastResult]:
        """
        Get results for a specific query
        
        Args:
            query_id: Query sequence ID
        
        Returns:
            BlastResult with all hits for the query, or None if not found
        """
        return self.results.get(query_id)
    
    def get_all_results(self) -> Dict[str, BlastResult]:
        """
        Get all results
        
        Returns:
            Dictionary mapping query_id -> BlastResult
        """
        return self.results
    
    def filter_by_evalue(self,
                        query_id: str,
                        evalue_cutoff: float) -> List[BlastHit]:
        """
        Filter hits by e-value threshold
        
        Args:
            query_id: Query sequence ID
            evalue_cutoff: E-value threshold
        
        Returns:
            Filtered list of BlastHit objects
        """
        result = self.get_results(query_id)
        if not result:
            return []
        
        return [hit for hit in result.hits if hit.evalue <= evalue_cutoff]
    
    def filter_by_identity(self,
                          query_id: str,
                          percent_identity_cutoff: float) -> List[BlastHit]:
        """
        Filter hits by percent identity
        
        Args:
            query_id: Query sequence ID
            percent_identity_cutoff: Minimum percent identity (0-100)
        
        Returns:
            Filtered list of BlastHit objects
        """
        result = self.get_results(query_id)
        if not result:
            return []
        
        return [hit for hit in result.hits
                if hit.percent_identity >= percent_identity_cutoff]
    
    def print_ground_truth_report(self,
                                 top_n: int = 50,
                                 evalue_cutoff: float = 0.01,
                                 max_queries_to_show: int = 5) -> None:
        """
        Print detailed Ground Truth report
        
        Useful for understanding BLAST parsing results before evaluation
        """
        print(f"\n=== BLAST Ground Truth Report ===")
        print(f"Parameters: E-value cutoff = {evalue_cutoff}, Top-N = {top_n}")
        print("-" * 70)
        
        stats = self.get_ground_truth_stats(top_n, evalue_cutoff)
        
        if not stats:
            print("No Ground Truth data available")
            return
        
        print(f"Total queries: {stats['num_queries']}")
        print(f"Total Ground Truth hits: {stats['total_ground_truth_hits']}")
        print(f"Average Ground Truth size per query: {stats['avg_ground_truth_size']:.1f}")
        print(f"Range: {stats['min_ground_truth_size']} - {stats['max_ground_truth_size']}")
        
        print(f"\nDetailed breakdown (first {max_queries_to_show} queries):")
        print("-" * 70)
        
        for i, (query_id, hit_list) in enumerate(self.results.items()):
            if i >= max_queries_to_show:
                remaining = len(self.results) - max_queries_to_show
                print(f"... and {remaining} more queries")
                break
            
            ground_truth_set = self.get_ground_truth_set(query_id, top_n, evalue_cutoff)
            top_hits = self.get_top_hits(query_id, top_n, evalue_cutoff)
            
            print(f"\n{i+1}. Query: {query_id}")
            print(f"   SBLAST size: |SBLAST| = {len(ground_truth_set)}")
            print(f"   Hits after E-value filter: {len(top_hits)}")
            
            # Show first 3 hits
            for j, hit in enumerate(top_hits[:3], 1):
                print(f"     {j}. {hit.subject_id}: bitscore={hit.bitscore:.1f}, "
                      f"evalue={hit.evalue:.2e}, identity={hit.percent_identity:.1f}%")
            
            if len(top_hits) > 3:
                print(f"     ... and {len(top_hits) - 3} more hits")
    
    def categorize_by_pident(self,
                            query_id: str,
                            top_n: int = 50,
                            evalue_cutoff: float = 0.01) -> dict:
        """
        Categorize filtered hits by pident (percent identity)
        
        Using BLAST column: pident
        
        Categories:
        - high_confidence: pident >= 30% (clear homology)
        - twilight_zone: 20% <= pident < 30% (remote homologs, BLAST weak signal)
        - below_detection: pident < 20% (BLAST fails, not typically in results)
        
        Args:
            query_id: Query sequence ID
            top_n: Top-N hits to categorize
            evalue_cutoff: E-value threshold
        
        Returns:
            Dictionary with categorized hits
        """
        top_hits = self.get_top_hits(query_id, top_n, evalue_cutoff)
        
        high_conf = []
        twilight = []
        below = []
        
        for hit in top_hits:
            if hit.percent_identity >= 30.0:
                high_conf.append(hit)
            elif hit.percent_identity >= 20.0:
                twilight.append(hit)
            else:
                below.append(hit)
        
        return {
            'query_id': query_id,
            'high_confidence': high_conf,      # pident >= 30%
            'twilight_zone': twilight,         # 20% <= pident < 30%
            'below_detection': below,          # pident < 20%
            'counts': {
                'high_confidence': len(high_conf),
                'twilight_zone': len(twilight),
                'below_detection': len(below),
            }
        }
    
    def get_ground_truth_stats(self,
                              top_n: int = 50,
                              evalue_cutoff: float = 0.01) -> dict:
        """
        Get statistics about Ground Truth sets across all queries
        
        Args:
            top_n: Maximum number of hits per query
            evalue_cutoff: E-value threshold for filtering
        
        Returns:
            Dictionary with Ground Truth statistics
        """
        ground_truth_sizes = []
        total_hits = 0
        
        for query_id in self.results.keys():
            gt_set = self.get_ground_truth_set(query_id, top_n, evalue_cutoff)
            ground_truth_sizes.append(len(gt_set))
            total_hits += len(gt_set)
        
        if not ground_truth_sizes:
            return {}
        
        return {
            'num_queries': len(ground_truth_sizes),
            'total_ground_truth_hits': total_hits,
            'avg_ground_truth_size': sum(ground_truth_sizes) / len(ground_truth_sizes),
            'min_ground_truth_size': min(ground_truth_sizes),
            'max_ground_truth_size': max(ground_truth_sizes),
        }
    
    def get_all_ground_truth_sets(self,
                                 top_n: int = 50,
                                 evalue_cutoff: float = 0.01) -> dict:
        """
        Get Ground Truth sets for all queries
        
        Returns dictionary: query_id -> SBLAST (set of subject IDs)
        
        Args:
            top_n: Maximum number of hits per query
            evalue_cutoff: E-value threshold
        
        Returns:
            Dictionary mapping query_id -> Ground Truth set
        """
        ground_truth = {}
        for query_id in self.results.keys():
            ground_truth[query_id] = self.get_ground_truth_set(query_id, top_n, evalue_cutoff)
        
        return ground_truth
    
    def filter_twilight_zone(self,
                            query_id: str,
                            lower_threshold: float = 20.0,
                            upper_threshold: float = 30.0) -> List[BlastHit]:
        """
        Filter hits in the Twilight Zone (20-30% identity)
        
        Twilight Zone characteristics:
        - BLAST shows weak signal (low identity)
        - Seed words fail to distinguish from noise
        - Many false negatives (missed remote homologs)
        - High E-values (results often rejected as noise)
        - Real remote homologs rank poorly
        
        Embedding methods (ESM-2/ANN) may succeed here because:
        - They capture structural/functional similarity
        - Not limited by sequence similarity threshold
        - Can detect conserved domains/patterns that BLAST misses
        
        Args:
            query_id: Query sequence ID
            lower_threshold: Lower bound (20% by default)
            upper_threshold: Upper bound (30% by default)
        
        Returns:
            List of BlastHit objects in the Twilight Zone
        """
        result = self.get_results(query_id)
        if not result:
            return []
        
        return [hit for hit in result.hits
                if lower_threshold <= hit.percent_identity < upper_threshold]
    
    def get_high_confidence_hits(self,
                                query_id: str,
                                identity_threshold: float = 30.0) -> List[BlastHit]:
        """
        Get high-confidence hits (>= 30% identity)
        
        At this identity level:
        - Clear indication of homology
        - BLAST reliably detects these relationships
        - Seed words effectively capture similarity
        - Generally low E-values
        
        Args:
            query_id: Query sequence ID
            identity_threshold: Identity threshold (30% by default)
        
        Returns:
            List of high-confidence BlastHit objects
        """
        result = self.get_results(query_id)
        if not result:
            return []
        
        return [hit for hit in result.hits
                if hit.percent_identity >= identity_threshold]
    
    def get_below_detection_hits(self,
                                query_id: str,
                                identity_threshold: float = 20.0) -> List[BlastHit]:
        """
        Get hits below BLAST detection threshold (< 20% identity)
        
        Below this threshold:
        - BLAST typically fails to detect (high false negative rate)
        - Cannot reliably distinguish homology from random similarity
        - These are potential "dark matter" of homology
        - Only detectible via structure/function (embedding methods)
        
        Note: Such hits may not appear in BLAST results due to filtering
        
        Args:
            query_id: Query sequence ID
            identity_threshold: Threshold below which detection fails (20% by default)
        
        Returns:
            List of BlastHit objects below detection threshold
        """
        result = self.get_results(query_id)
        if not result:
            return []
        
        return [hit for hit in result.hits
                if hit.percent_identity < identity_threshold]
    
    def get_twilight_zone_stats(self,
                               query_id: str) -> dict:
        """
        Get statistics about Twilight Zone hits for a query
        
        Useful for analyzing BLAST limitations and remote homolog potential
        
        Returns:
            Dictionary with Twilight Zone analysis
        """
        result = self.get_results(query_id)
        if not result:
            return {}
        
        high_conf = self.get_high_confidence_hits(query_id)
        twilight = self.filter_twilight_zone(query_id)
        below = self.get_below_detection_hits(query_id)
        
        return {
            'query_id': query_id,
            'total_hits': len(result.hits),
            'high_confidence_count': len(high_conf),
            'twilight_zone_count': len(twilight),
            'below_detection_count': len(below),
            'high_confidence_pct': 100 * len(high_conf) / len(result.hits) if result.hits else 0,
            'twilight_zone_pct': 100 * len(twilight) / len(result.hits) if result.hits else 0,
            'below_detection_pct': 100 * len(below) / len(result.hits) if result.hits else 0,
        }
    
    def get_homologs(self,
                    query_id: str,
                    evalue_cutoff: Optional[float] = None) -> List[str]:
        """
        Get all homologs (subject IDs) for a query
        
        Args:
            query_id: Query sequence ID
            evalue_cutoff: Optional E-value threshold (uses default if None)
        
        Returns:
            List of subject IDs representing homologs
        """
        if evalue_cutoff is None:
            evalue_cutoff = self.evalue_threshold
        
        hits = self.filter_by_evalue(query_id, evalue_cutoff)
        return [hit.subject_id for hit in hits]
    
    def get_top_hits(self,
                    query_id: str,
                    top_n: int = 50,
                    evalue_cutoff: float = 0.01) -> List[BlastHit]:
        """
        Get top N hits for a query with noise filtering
        
        BLAST Parsing & Filtering Pipeline - Using 3 key BLAST columns:
        
        [Step 1] COLUMN: evalue
                 Filter: Remove hits with E-value > 0.01
                 Removes noise and unreliable matches
                 
        [Step 2] COLUMN: bitscore  
                 Sort: Descending order (higher = better quality)
                 Provides Top-N ranking by alignment quality
                 
        [Step 3] COLUMN: pident (percent identity)
                 For downstream analysis:
                 - pident >= 30%: High-confidence homologs
                 - 20% <= pident < 30%: Twilight Zone (remote homologs)
                 - pident < 20%: Below BLAST detection threshold
        
        Output: Top-N filtered & sorted hits that form Ground Truth: SBLAST
        
        Args:
            query_id: Query sequence ID
            top_n: Number of top hits to return
            evalue_cutoff: E-value threshold (default: 0.01)
        
        Returns:
            List of top BlastHit objects sorted by bitscore (descending)
        """
        result = self.get_results(query_id)
        if not result:
            return []
        
        # Step 1: Filter by evalue (remove E-value > cutoff)
        # This removes noise: keeps only statistically significant hits
        filtered_hits = [hit for hit in result.hits if hit.evalue <= evalue_cutoff]
        
        if not filtered_hits:
            print(f"Warning: No hits pass E-value filter ({evalue_cutoff}) for {query_id}")
            return []
        
        # Step 2: Sort by bitscore (descending) - higher bitscore = better alignment
        # This provides the Top-N quality ranking
        sorted_hits = sorted(filtered_hits, key=lambda h: -h.bitscore)
        
        # Step 3: Select top-N hits
        return sorted_hits[:top_n]
    
    def get_top_subject_ids(self,
                           query_id: str,
                           top_n: int = 50,
                           evalue_cutoff: float = 0.01) -> List[str]:
        """
        Get top N subject IDs for a query (for Recall@N calculation)
        
        Returns filtered and sorted subject IDs that form the Ground Truth set:
        SBLAST = {id1, id2, ..., idk} where k ≤ N
        
        Args:
            query_id: Query sequence ID
            top_n: Number of top hits to return
            evalue_cutoff: E-value threshold (default: 0.01)
        
        Returns:
            List of top subject IDs
        """
        return [hit.subject_id for hit in self.get_top_hits(query_id, top_n, evalue_cutoff)]
    
    def get_ground_truth_set(self,
                            query_id: str,
                            top_n: int = 50,
                            evalue_cutoff: float = 0.01) -> set:
        """
        Get Ground Truth set for a query (SBLAST)
        
        BLAST Parsing Output:
        SBLAST(query_i) = {subject_id_1, subject_id_2, ..., subject_id_k}
        
        where:
        - Each subject_id is a "true" neighbor
        - Filtered by E-value > evalue_cutoff (removes noise)
        - Top-N ranked by Bit-Score (quality ranking)
        - k ≤ N (may be fewer if not enough high-quality hits)
        
        Args:
            query_id: Query sequence ID
            top_n: Maximum number of hits (N in Top-N)
            evalue_cutoff: E-value threshold for noise filtering (default: 0.01)
        
        Returns:
            Set of subject IDs (SBLAST for this query)
        """
        return set(self.get_top_subject_ids(query_id, top_n, evalue_cutoff))
    
    def save_results(self,
                    filepath: str,
                    format: str = "json") -> bool:
        """
        Save results to file
        
        Args:
            filepath: Output file path
            format: Output format ("json" or "tsv")
        
        Returns:
            True if save successful
        """
        try:
            if format == "json":
                data = {
                    query_id: {
                        'num_hits': len(result.hits),
                        'hits': [hit.to_dict() for hit in result.hits]
                    }
                    for query_id, result in self.results.items()
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == "tsv":
                with open(filepath, 'w') as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=self.BLAST_COLUMNS,
                        delimiter='\t'
                    )
                    writer.writeheader()
                    
                    for result in self.results.values():
                        for hit in result.hits:
                            writer.writerow(hit.to_dict())
            
            else:
                print(f"Unsupported format: {format}")
                return False
            
            print(f"Results saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def load_results(self, filepath: str) -> bool:
        """
        Load results from file
        
        Args:
            filepath: Input file path
        
        Returns:
            True if load successful
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return False
        
        try:
            # Try to load as JSON first
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.results.clear()
                    
                    for query_id, result_data in data.items():
                        hits = [
                            BlastHit(**hit_data)
                            for hit_data in result_data['hits']
                        ]
                        self.results[query_id] = BlastResult(
                            query_id=query_id,
                            hits=hits
                        )
            
            else:  # Assume TSV format
                self.output_path = filepath
                return self.parse_output()
            
            print(f"Results loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def clear_results(self) -> None:
        """Clear all results"""
        self.results.clear()
    
    def get_num_queries(self) -> int:
        """Get number of queries processed"""
        return len(self.results)
    
    def get_execution_time(self) -> Optional[float]:
        """
        Get total BLAST execution time in seconds
        
        Returns:
            Total execution time, or None if search hasn't run yet
        """
        return self.execution_time
    
    def get_time_per_query(self) -> Optional[float]:
        """
        Get average time per query
        
        For performance evaluation table:
        Time/query (s) = Total execution time / Number of queries
        
        Returns:
            Time per query in seconds, or None if no queries
        """
        if self.execution_time is None or self.get_num_queries() == 0:
            return None
        
        return self.execution_time / self.get_num_queries()
    
    def get_queries_per_second(self) -> Optional[float]:
        """
        Get throughput: queries per second (QPS)
        
        For performance evaluation table:
        QPS = Number of queries / Total execution time
        
        Returns:
            Queries per second, or None if no execution time
        """
        if self.execution_time is None or self.execution_time == 0:
            return None
        
        return self.get_num_queries() / self.execution_time
    
    def print_summary(self) -> None:
        """Print results summary with timing information"""
        print("\n=== BLAST Results Summary ===")
        print(f"Queries processed: {len(self.results)}")
        
        # Print timing metrics
        if self.execution_time is not None:
            print(f"\nTiming Metrics:")
            print(f"  Total execution time: {self.execution_time:.2f}s")
            print(f"  Time/query: {self.get_time_per_query():.3f}s")
            print(f"  Throughput (QPS): {self.get_queries_per_second():.2f} queries/sec")
        
        # Print hit statistics
        print(f"\nHit Statistics:")
        for query_id, result in list(self.results.items())[:5]:  # Show first 5 queries
            print(f"\nQuery: {query_id}")
            print(f"  Number of hits: {len(result.hits)}")
            
            if result.hits:
                best_hit = min(result.hits, key=lambda h: h.evalue)
                print(f"  Best hit: {best_hit.subject_id}")
                print(f"    E-value: {best_hit.evalue:.2e}")
                print(f"    Identity: {best_hit.percent_identity:.1f}%")
                print(f"    Bit score: {best_hit.bitscore:.1f}")
        
        if len(self.results) > 5:
            print(f"\n... and {len(self.results) - 5} more queries")
    
    def print_twilight_analysis(self, query_id: str) -> None:
        """
        Print detailed Twilight Zone analysis for a query
        Useful for understanding BLAST limitations and embedding method opportunities
        """
        stats = self.get_twilight_zone_stats(query_id)
        
        if not stats:
            print(f"No results for query {query_id}")
            return
        
        print(f"\n=== Twilight Zone Analysis: {query_id} ===")
        print(f"Total BLAST hits: {stats['total_hits']}")
        print(f"\nClassification breakdown:")
        print(f"  High-confidence (>= 30%): {stats['high_confidence_count']} ({stats['high_confidence_pct']:.1f}%)")
        print(f"    → Clear homology signal, BLAST reliable")
        print(f"  Twilight Zone (20-30%):   {stats['twilight_zone_count']} ({stats['twilight_zone_pct']:.1f}%)")
        print(f"    → Remote homologs, BLAST weak/missing signal")
        print(f"    → Embedding methods may detect (structure/function preserved)")
        print(f"  Below detection (< 20%):  {stats['below_detection_count']} ({stats['below_detection_pct']:.1f}%)")
        print(f"    → BLAST typically fails, cannot distinguish from noise")
        print(f"    → Potential 'dark matter' only detectable via embedding")
        
        # Show Twilight Zone hits as examples
        twilight = self.filter_twilight_zone(query_id)
        if twilight:
            print(f"\nTwilight Zone hits ({len(twilight)} total):")
            for i, hit in enumerate(twilight[:5], 1):  # Show top 5
                print(f"  {i}. {hit.subject_id}: {hit.percent_identity:.1f}% identity, "
                      f"E={hit.evalue:.2e}, Bits={hit.bitscore:.1f}")
            if len(twilight) > 5:
                print(f"  ... and {len(twilight) - 5} more")


# Example usage
if __name__ == "__main__":
    # Example parameters (modify paths as needed)
    blastp_path = "blastp"                    # blastp executable
    makeblastdb_path = "makeblastdb"          # makeblastdb executable
    swissprot_fasta = "./data/swissprot.fasta"
    database_name = "./data/swissprot_db"
    queries_fasta = "./data/targets.fasta"
    
    print("=== BLAST Ground Truth Pipeline ===\n")
    
    # Step 1: Create BLAST database
    print("Step 1: Database Creation")
    print("-" * 50)
    if not Blast.create_database(swissprot_fasta, database_name, makeblastdb_path):
        print("Failed to create database")
        exit(1)
    
    # Step 2: Initialize BLAST wrapper
    print("\nStep 2: BLAST Search Setup")
    print("-" * 50)
    blast = Blast(blastp_path, database_name, queries_fasta)
    blast.set_parameters(evalue=1e-5, max_targets=250, threads=4)
    
    # Step 3: Run search
    print("\nStep 3: BLAST Execution")
    print("-" * 50)
    if not blast.run_search():
        print("BLAST search failed")
        exit(1)
    
    # Step 4: Analyze results
    print("\nStep 4: Results Analysis")
    print("-" * 50)
    blast.print_summary()
    
    # Step 5: Twilight Zone analysis
    print("\nStep 5: Twilight Zone Analysis (Remote Homolog Detection)")
    print("-" * 50)
    if blast.get_num_queries() > 0:
        first_query = list(blast.get_all_results().keys())[0]
        blast.print_twilight_analysis(first_query)
        
        # Save results
        blast.save_results("./blast_results.json", format="json")
        blast.save_results("./blast_results.tsv", format="tsv")
        print(f"\nResults saved to:")
        print(f"  - blast_results.json")
        print(f"  - blast_results.tsv")
