import json
import numpy as np
from scipy.stats import ttest_ind
import argparse

def compute_statistics(metrics_opl, metrics_nl):
    # Include all metrics for evaluation
    metrics = ['loose', 'strict', 'rouge1', 'rouge2', 'rougeL', 'bleurt', 'gpt_judge', 'precision', 'recall', 'f1']
    results = {}
    for metric in metrics:
        # Extract values for the metric
        values_opl = [item['metrics'][metric] for item in metrics_opl if metric in item['metrics']]
        values_nl = [item['metrics'][metric] for item in metrics_nl if metric in item['metrics']]
        
        # Compute mean and standard deviation
        mean_opl = np.mean(values_opl)
        std_opl = np.std(values_opl, ddof=1)
        mean_nl = np.mean(values_nl)
        std_nl = np.std(values_nl, ddof=1)
        
        # Perform a two-sample t-test
        t_stat, p_value = ttest_ind(values_opl, values_nl, equal_var=False)
        
        # Store results
        results[metric] = {
            'mean_opl': mean_opl,
            'std_opl': std_opl,
            'mean_nl': mean_nl,
            'std_nl': std_nl,
            'p_value': p_value
        }
    return results

def main(eval_opl_file, eval_nl_file, output_file):
    # Load the evaluation data
    with open(eval_opl_file, 'r') as f_opl, open(eval_nl_file, 'r') as f_nl:
        data_opl = json.load(f_opl)
        data_nl = json.load(f_nl)
    
    details_opl = data_opl['details']
    details_nl = data_nl['details']
    
    # Compute statistics
    results = compute_statistics(details_opl, details_nl)
    
    # Write results to the output file
    with open(output_file, 'w') as f:
        for metric, stats in results.items():
            f.write(f"{metric}:\n")
            f.write(f"  OPL - Mean: {stats['mean_opl']:.3f}, Std: {stats['std_opl']:.3f}\n")
            f.write(f"  NL  - Mean: {stats['mean_nl']:.3f}, Std: {stats['std_nl']:.3f}\n")
            f.write(f"  P-value: {stats['p_value']:.3e}\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics and p-values for two QA evaluation metrics files.")
    parser.add_argument('--eval_opl', type=str, required=True, help="Evaluation JSON file for the 'opl' experiment")
    parser.add_argument('--eval_nl', type=str, required=True, help="Evaluation JSON file for the 'nl' experiment")
    parser.add_argument('--output', type=str, required=True, help="Output TXT file")
    args = parser.parse_args()
    main(args.eval_opl, args.eval_nl, args.output)
