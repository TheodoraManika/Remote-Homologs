from typing import List, Dict, Optional

class ReportGenerator:
    """
    Generates formatted reports for the protein homology search experiment.
    Matches the specified output format with Greek/English headers as requested.
    """

    @staticmethod
    def generate_report(
        query_id: str,
        n: int,
        methods_stats: List[Dict],
        blast_ground_truth: set,
        blast_hits_details: Dict[str, float], # Map subject_id -> identity
        pfam_map: Dict[str, str], # Map protein_id -> Pfam ID
        top_neighbors: Dict[str, List[Dict]], # Map method_name -> list of neighbor dicts {id, dist, ...}
    ):
        """
        Prints the full report for a single query.
        
        Args:
            query_id: The ID of the query protein.
            n: Top-N size (e.g., 50).
            methods_stats: List of dicts with 'name', 'time', 'qps', 'recall'.
            blast_ground_truth: Set of IDs in BLAST Top-N (S_BLAST).
            blast_hits_details: Dict mapping subject_id to BLAST identity %.
            pfam_map: Dict mapping protein IDs to Pfam annotations.
            top_neighbors: Dict mapping method name to list of neighbors.
        """
        
        print(f"Query Protein: {query_id}")
        print(f"N = {n} (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)")
        
        # [1] Summary Table
        print("\n[1] Συνοπτική σύγκριση μεθόδων")
        print("-" * 70)
        print(f"{'Method':<20} | {'Time/query (s)':<15} | {'QPS':<10} | {'Recall@N vs BLAST Top-N':<25}")
        print("-" * 70)
        
        for method in methods_stats:
            name = method.get('name', 'Unknown')
            t = method.get('time_per_query', 0.0)
            q = method.get('qps', 0.0)
            rec = method.get('recall', 0.0)
            print(f"{name:<20} | {t:<15.4f} | {q:<10.1f} | {rec:<25.4f}")
            
        print("-" * 70)
        
        # [2] Detailed Neighbors Table
        print(f"\n[2] Top-N γείτονες ανά μέθοδο (εδώ π.χ. N = {min(10, n)} για εκτύπωση)")
        
        query_pfam = pfam_map.get(query_id, "Unknown")
        
        for method_name, neighbors in top_neighbors.items():
            print(f"\nMethod: {method_name}")
            print(f"{'Rank':<5} | {'Neighbor ID':<15} | {'L2 Dist':<10} | {'BLAST Identity':<15} | {'In BLAST Top-N?':<15} | {'Bio comment'}")
            print("-" * 90)
            
            # Show only top 10 for detailed view as per example
            for rank, neighbor in enumerate(neighbors[:10], 1):
                nid = neighbor.get('id', 'N/A')
                dist = neighbor.get('distance', 0.0)
                
                # Check BLAST info
                in_blast = "Yes" if nid in blast_ground_truth else "No"
                identity = f"{blast_hits_details.get(nid, 0.0):.1f}%" if nid in blast_hits_details else "N/A"
                if nid in blast_hits_details and blast_hits_details[nid] == 0.0:
                     identity = "N/A" # Distinguish 0% from missing if needed, or just 0.0%

                # Bio Comment Logic
                subject_pfam = pfam_map.get(nid, "Unknown")
                comment = "--"
                
                if in_blast == "Yes" and subject_pfam != "Unknown" and subject_pfam == query_pfam:
                     comment = f"Remote homolog? ({subject_pfam})"
                elif in_blast == "No":
                    if subject_pfam != "Unknown" and subject_pfam == query_pfam:
                         comment = "Potential Remote Homolog (New)"
                    else:
                         comment = "Possible false positive"
                
                print(f"{rank:<5} | {nid:<15} | {dist:<10.4f} | {identity:<15} | {in_blast:<15} | {comment}")
