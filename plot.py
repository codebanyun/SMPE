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

def main():
    base_dir = "/root/research/results/sacred/smpe/Foraging-2s-9x9-3p-2f-coop-v2"
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Runs 35 to 42 (inclusive)
    run_ids = range(35, 43) 

    for run_id in run_ids:
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

if __name__ == "__main__":
    main()
