import os
import subprocess
import sys

def run_analysis():
    """Run both analysis scripts in sequence"""
    print("\n=== Running Keyword Frequency Analysis ===\n")
    
    # Run the main analysis script
    try:
        subprocess.run([sys.executable, "keyword_analysis.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error running keyword analysis script")
        return
        
    print("\n=== Running Additional Visualizations ===\n")
    
    # Run the visualizations script
    try:
        subprocess.run([sys.executable, "sector_visualizations.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error running visualizations script")
        
    print("\n=== Analysis Complete ===\n")
    
    # List all result directories
    results_dir = 'results'
    if os.path.exists(results_dir):
        subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        subdirs.sort(reverse=True)  # Most recent first
        
        print(f"Available result directories (most recent first):")
        for i, d in enumerate(subdirs[:5]):  # Show the 5 most recent
            print(f"  {i+1}. {d}")

if __name__ == "__main__":
    run_analysis()
