import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_run(metrics_path, run_name, out_path, metric_name="test_battle_won_mean"):
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {metrics_path}: {e}")
        return

    if metric_name not in data:
        print(f"Metric {metric_name} not found in {metrics_path}")
        return

    steps = np.array(data[metric_name]["steps"], dtype=np.float64)
    values = np.array(data[metric_name]["values"], dtype=np.float64)
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, values, linewidth=1.5)
    plt.xlabel("t_env")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.title(f"{run_name}: {metric_name}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path}")

def plot_comparison(runs, out_path, metric_name="test_battle_won_mean"):
    plt.figure(figsize=(8, 4.5))
    
    for metrics_path, run_name in runs:
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {metrics_path}: {e}")
            continue

        if metric_name not in data:
            print(f"Metric {metric_name} not found in {metrics_path}")
            continue

        steps = np.array(data[metric_name]["steps"], dtype=np.float64)
        values = np.array(data[metric_name]["values"], dtype=np.float64)
        
        plt.plot(steps, values, linewidth=1.5, label=run_name)

    plt.xlabel("t_env")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.title(f"Comparison: {metric_name}")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path}")

def main():
    runs = [
        ("/root/research/results/sacred/smpe/3m/6/metrics.json", "Run_6"),
        ("/root/research/results/sacred/smpe/3m/9/metrics.json", "Run_9")
    ]
    
    out_dir = "/root/research/smpe_raw/plot"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for path, name in runs:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        filename = f"{name}_test_battle_won_mean.png"
        out_path = os.path.join(out_dir, filename)
        plot_run(path, name, out_path)

    # Plot comparison
    comparison_out_path = os.path.join(out_dir, "Run_6_vs_Run_9_test_battle_won_mean.png")
    plot_comparison(runs, comparison_out_path)

if __name__ == "__main__":
    main()
