import os
import subprocess
import time
import re
import shutil
import sys

def run_command(command, cwd=None):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return False
    # Only print stdout if it's not too verbose or if important
    # print(result.stdout) 
    print("Success")
    return True

def compile_cpp():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cpp_dir = os.path.join(base_dir, "implementations/Similarity-Search")
    
    print("\n[Compiling C++ Search Algorithms]")
    
    # Check if Makefile exists
    makefile_path = os.path.join(cpp_dir, "Makefile")
    if not os.path.exists(makefile_path):
        print(f"Error: Makefile not found at {makefile_path}")
        return False

    # Clean previous build
    if not run_command("make clean", cwd=cpp_dir):
        print("Warning: Clean failed (might be first run)")
    
    # Run make
    if not run_command("make", cwd=cpp_dir):
        print("Error: Compilation failed")
        return False
        
    # Check if executable exists
    executable_path = os.path.join(cpp_dir, "search")
    if not os.path.exists(executable_path):
        print(f"Error: Executable 'search' not found after compilation at {executable_path}")
        return False
        
    # Copy executable to root for convenience (optional, but good for handler)
    root_executable = os.path.join(base_dir, "search")
    shutil.copy2(executable_path, root_executable)
    print(f"Copied executable to {root_executable}")
    
    return True

class AlgoHandler:
    def __init__(self, executable_path="./search", output_dir="results"):
        self.executable = executable_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _run_search(self, args_list, output_file):
        """Helper to execute the C++ binary with arguments."""
        cmd = [self.executable] + args_list
        # Output file path handling
        # The C++ interface handles -o by putting it in results/, but let's be explicit
        if not output_file.startswith("results/"):
            output_path = os.path.join(self.output_dir, output_file)
        else:
            output_path = output_file
            
        cmd.extend(["-o", output_path])
        
        cmd_str = " ".join(cmd)
        print(f"Executing: {cmd_str}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            print(f"Error running {cmd_str}")
            print(result.stderr)
            return None
            
        # Parse metrics from stdout or return the output file path for further parsing
        return {
            "output_file": output_path,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "wall_time": end_time - start_time
        }

    def run_lsh(self, input_file, query_file, output_file, k=4, L=5, w=4.0, N=1, R=10000.0):
        args = [
            "-d", input_file,
            "-q", query_file,
            "-type", "sift", # Using sift type for fvecs format
            "-lsh",
            "-k", str(k),
            "-L", str(L),
            "-w", str(w),
            "-N", str(N),
            "-R", str(R)
        ]
        return self._run_search(args, output_file)

    def run_hypercube(self, input_file, query_file, output_file, kproj=14, M=500, probes=2, w=4.0, N=1, R=10000.0):
        args = [
            "-d", input_file,
            "-q", query_file,
            "-type", "sift",
            "-hypercube",
            "-kproj", str(kproj),
            "-M", str(M),
            "-probes", str(probes),
            "-w", str(w),
            "-N", str(N),
            "-R", str(R)
        ]
        return self._run_search(args, output_file)

    def run_ivfflat(self, input_file, query_file, output_file, kclusters=50, nprobe=5, N=1, R=10000.0):
        args = [
            "-d", input_file,
            "-q", query_file,
            "-type", "sift",
            "-ivfflat",
            "-kclusters", str(kclusters),
            "-nprobe", str(nprobe),
            "-N", str(N),
            "-R", str(R)
        ]
        return self._run_search(args, output_file)

    def run_ivfpq(self, input_file, query_file, output_file, kclusters=50, nprobe=5, M=8, nbits=8, N=1, R=10000.0):
        args = [
            "-d", input_file,
            "-q", query_file,
            "-type", "sift",
            "-ivfpq",
            "-kclusters", str(kclusters),
            "-nprobe", str(nprobe),
            "-M", str(M),
            "-nbits", str(nbits),
            "-N", str(N),
            "-R", str(R)
        ]
        return self._run_search(args, output_file)

    def parse_query_results(self, output_file):
        """
        Parses the detailed neighbor results from the output file.
        Returns a dict: { query_index (int): [ {'id': neighbor_index (int), 'distance': float}, ... ] }
        """
        results = {}
        current_query = None
        current_neighbors = []
        
        if not os.path.exists(output_file):
            return {}

        with open(output_file, 'r') as f:
            lines = f.readlines()
            
        # Format usually:
        # Query: 0
        # Nearest neighbor-1: 100
        # distanceApproximate: 0.123
        # ...
        
        # Regex patterns
        query_pat = re.compile(r"Query:\s+(\d+)")
        neighbor_pat = re.compile(r"Nearest neighbor-\d+:\s+(\d+)")
        dist_pat = re.compile(r"distanceApproximate:\s+([\d\.]+)")
        
        temp_neighbor_id = None
        
        for line in lines:
            line = line.strip()
            
            # Check for new query
            q_match = query_pat.match(line)
            if q_match:
                # Save previous query if exists
                if current_query is not None:
                    results[current_query] = current_neighbors
                
                current_query = int(q_match.group(1))
                current_neighbors = []
                temp_neighbor_id = None
                continue
            
            # Check for neighbor ID
            n_match = neighbor_pat.match(line)
            if n_match:
                temp_neighbor_id = int(n_match.group(1))
                continue
                
            # Check for distance (comes after neighbor ID)
            d_match = dist_pat.match(line)
            if d_match and temp_neighbor_id is not None:
                dist = float(d_match.group(1))
                current_neighbors.append({
                    'id': temp_neighbor_id,
                    'distance': dist
                })
                temp_neighbor_id = None # Reset
                
        # Save last query
        if current_query is not None:
            results[current_query] = current_neighbors
            
        return results

    def parse_metrics(self, result_dict):
        """
        Parses the stdout or output file to extract QPS, Recall, etc.
        Based on Interface.cpp, it calls printFinalStatistics.
        """
        if not result_dict or not result_dict.get("output_file"):
            return {}
            
        metrics = {}
        # Try parsing stdout first as Interface.cpp might print summaries there?
        # Actually Interface.cpp prints to the output file mostly.
        # Let's read the output file's last lines for statistics.
        
        try:
            with open(result_dict["output_file"], 'r') as f:
                content = f.read()
                
            # Look for patterns like "Average Recall: 0.95" etc.
            # Adjust these regexes based on exact C++ output format seen in Interface.cpp
            # printFinalStatistics output:
            # "Approximation Factor: <af>"
            # "Recall: <recall>"
            # "QPS: <qps>"
            
            af_match = re.search(r"Approximation Factor: ([\d\.]+)", content)
            recall_match = re.search(r"Recall: ([\d\.]+)", content)
            # MAF is commonly printed as "MAF: ..." or "Max Approximation Factor"
            
            # QPS might be printed to stdout in the previous step?
            # Interface.cpp: printFinalStatistics(output, ...);
            # Let's assume the helper `printFinalStatistics` writes key-value pairs at the end.
            
            # From Interface.cpp:
            # output << "Approximation Factor: " << avg_af << std::endl;
            # output << "Recall: " << avg_recall << std::endl;
            # output << "QPS: " << qps << std::endl;
            
            if af_match: metrics['af'] = float(af_match.group(1))
            if recall_match: metrics['recall'] = float(recall_match.group(1))
            
            qps_match = re.search(r"QPS: ([\d\.]+)", content)
            if qps_match: metrics['qps'] = float(qps_match.group(1))
            
            time_match = re.search(r"Average Approximate Time: ([\d\.]+)", content)
            if time_match: metrics['time_per_query'] = float(time_match.group(1))

        except Exception as e:
            print(f"Error parsing metrics: {e}")
            
        return metrics

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        if compile_cpp():
            print("\nCompilation successful!")
        else:
            print("\nCompilation failed!")
            sys.exit(1)
    else:
        print("AlgoHandler module.")
        print("Run 'python3 algo_handler/handler.py compile' to compile the C++ backend.")
