import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _load_metric(metrics_json_path, metric_name):
    try:
        with open(metrics_json_path, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Error reading {metrics_json_path}: {e}")
        return None, None

    if metric_name not in metrics:
        print(f"Warning: Metric '{metric_name}' not found in {metrics_json_path}")
        return None, None
    
    steps = np.asarray(metrics[metric_name]["steps"], dtype=np.float64)
    values = np.asarray(metrics[metric_name]["values"], dtype=np.float64)
    return steps, values

def _plot_metric(run_id, metrics_json_path, metric_name, out_path):
    steps, values = _load_metric(metrics_json_path, metric_name)
    if steps is None:
        return

    x = steps / 1e7

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, values, linewidth=1.5)
    plt.xlabel("t_env (x1e7)")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.title(f"Run {run_id}: {metric_name}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path}")

def _plot_comparison(run_ids, base_dir, metric_name, out_path):
    plt.figure(figsize=(8, 4.5))
    
    for run_id in run_ids:
        metrics_json_path = os.path.join(base_dir, str(run_id), "metrics.json")
        if not os.path.exists(metrics_json_path):
            print(f"Skipping run {run_id} for comparison, metrics.json not found")
            continue
            
        steps, values = _load_metric(metrics_json_path, metric_name)
        if steps is not None:
            x = steps / 1e7
            plt.plot(x, values, label=f"Run {run_id}", linewidth=1.5)

    plt.xlabel("t_env (x1e7)")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.title(f"Comparison: {metric_name}")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved comparison plot to {out_path}")

def main():
    base_dir = "/root/research/results/sacred/smpe/Foraging-2s-9x9-3p-2f-coop-v2"
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Individual plots for specific runs
    target_runs = [18, 19, 55, 57, 58, 59]
    for run_id in target_runs:
        run_path = os.path.join(base_dir, str(run_id), "metrics.json")
        if not os.path.exists(run_path):
            print(f"Skipping run {run_id}, metrics.json not found at {run_path}")
            continue
        
        # Plot test_return_mean
        _plot_metric(run_id, run_path, "test_return_mean", 
                     os.path.join(out_dir, f"run{run_id}_test_return_mean.png"))
        
        # Plot return_mean
        _plot_metric(run_id, run_path, "return_mean", 
                     os.path.join(out_dir, f"run{run_id}_return_mean.png"))

    # Comparison plot: 55 vs 12
    _plot_comparison([12, 55], base_dir, "test_return_mean",
                     os.path.join(out_dir, "comparison_run12_run55_test_return_mean.png"))
    
    # Comparison plot: 57 vs 12
    _plot_comparison([12, 57], base_dir, "test_return_mean",
                     os.path.join(out_dir, "comparison_run12_run57_test_return_mean.png"))

    # Comparison plot: 12 vs 21
    _plot_comparison([12, 21], base_dir, "test_return_mean",
                     os.path.join(out_dir, "comparison_run12_run21_test_return_mean.png"))

    # Comparison plot: 12 vs 18
    _plot_comparison([12, 18], base_dir, "test_return_mean",
                     os.path.join(out_dir, "comparison_run12_run18_test_return_mean.png"))

    # Comparison plot: 12 vs 19
    _plot_comparison([12, 19], base_dir, "test_return_mean",
                     os.path.join(out_dir, "comparison_run12_run19_test_return_mean.png"))

    # Comparison plot: 12 vs 59
    _plot_comparison([12, 59], base_dir, "test_return_mean",
                     os.path.join(out_dir, "comparison_run12_run59_test_return_mean.png"))

    # Comparison plot: 57 vs 59
    _plot_comparison([57, 59], base_dir, "test_return_mean",
                     os.path.join(out_dir, "comparison_run57_run59_test_return_mean.png"))
    
    # Comparison plot: 12, 55, 57
    _plot_comparison([12, 55, 57], base_dir, "test_return_mean",
                     os.path.join(out_dir, "comparison_run12_55_57_test_return_mean.png"))

if __name__ == "__main__":
    main()
