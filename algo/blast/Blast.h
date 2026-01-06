#pragma once

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <memory>

/**
 * BLAST Hit Result Structure
 * Represents a single alignment/hit from BLAST
 */
struct BlastHit {
    std::string query_id;           // Query sequence ID
    std::string subject_id;         // Subject (database) sequence ID
    double percent_identity;        // Percent identity
    int alignment_length;           // Length of alignment
    int mismatches;                 // Number of mismatches
    int gap_opens;                  // Number of gap openings
    int query_start;                // Start position in query
    int query_end;                  // End position in query
    int subject_start;              // Start position in subject
    int subject_end;                // End position in subject
    double evalue;                  // E-value (expect value)
    double bitscore;                // Bit score
    
    BlastHit() : percent_identity(0), alignment_length(0), mismatches(0),
                 gap_opens(0), query_start(0), query_end(0),
                 subject_start(0), subject_end(0), evalue(0), bitscore(0) {}
};

/**
 * BLAST Result Collection
 * Stores all hits for a query sequence
 */
struct BlastResult {
    std::string query_id;
    std::vector<BlastHit> hits;
    int num_hits;
    
    BlastResult() : num_hits(0) {}
};

/**
 * BLAST Algorithm Wrapper
 * Wraps NCBI blastp executable for protein sequence similarity search
 */
class Blast {
private:
    std::string blastp_path;        // Path to blastp executable
    std::string database_path;      // Path to formatted database (swissprot)
    std::string queries_path;       // Path to queries FASTA file
    std::string output_path;        // Path to output file
    
    // BLAST search parameters
    double evalue_threshold;        // E-value threshold for filtering
    int max_target_seqs;            // Maximum number of target sequences
    int num_threads;                // Number of threads to use
    
    // Results storage
    std::map<std::string, BlastResult> results;
    
public:
    /**
     * Constructor
     * @param blastp_path Path to blastp executable
     * @param database_path Path to formatted BLAST database
     * @param queries_path Path to queries FASTA file
     * @param output_path Path to output file
     */
    Blast(const std::string& blastp_path,
          const std::string& database_path,
          const std::string& queries_path,
          const std::string& output_path = "./blast_output.tsv");
    
    ~Blast();
    
    /**
     * Set BLAST search parameters
     * @param evalue E-value threshold
     * @param max_targets Maximum number of hits to report
     * @param threads Number of threads
     */
    void set_parameters(double evalue = 1e-5, int max_targets = 250, int threads = 4);
    
    /**
     * Execute BLAST search
     * Calls blastp executable with specified parameters
     * @return true if search completed successfully
     */
    bool run_search();
    
    /**
     * Parse BLAST output in TSV format
     * @return true if parsing successful
     */
    bool parse_output();
    
    /**
     * Get results for a specific query
     * @param query_id Query sequence ID
     * @return BlastResult with all hits for the query
     */
    BlastResult get_results(const std::string& query_id) const;
    
    /**
     * Get all results
     * @return Map of query_id -> BlastResult
     */
    const std::map<std::string, BlastResult>& get_all_results() const;
    
    /**
     * Filter results by e-value
     * @param query_id Query sequence ID
     * @param evalue_cutoff E-value threshold
     * @return Filtered hits
     */
    std::vector<BlastHit> filter_by_evalue(const std::string& query_id, 
                                           double evalue_cutoff) const;
    
    /**
     * Filter results by percent identity
     * @param query_id Query sequence ID
     * @param percent_identity_cutoff Minimum percent identity
     * @return Filtered hits
     */
    std::vector<BlastHit> filter_by_identity(const std::string& query_id,
                                            double percent_identity_cutoff) const;
    
    /**
     * Get homologs for a query (all significant hits)
     * @param query_id Query sequence ID
     * @return Vector of subject IDs representing homologs
     */
    std::vector<std::string> get_homologs(const std::string& query_id) const;
    
    /**
     * Save results to file
     * @param filepath Output file path
     * @param format Output format ("tsv", "json")
     * @return true if save successful
     */
    bool save_results(const std::string& filepath, const std::string& format = "tsv") const;
    
    /**
     * Load results from file
     * @param filepath Input file path
     * @return true if load successful
     */
    bool load_results(const std::string& filepath);
    
    /**
     * Clear all results
     */
    void clear_results();
    
    /**
     * Get number of queries processed
     * @return Number of queries
     */
    size_t get_num_queries() const { return results.size(); }
    
    /**
     * Print results summary
     */
    void print_summary() const;
};
