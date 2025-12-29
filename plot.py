import json
import os

import numpy as np


def _load_metric(metrics_json_path, metric_name):
    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)
    if metric_name not in metrics:
        raise KeyError(f"Metric '{metric_name}' not found in {metrics_json_path}")
    steps = np.asarray(metrics[metric_name]["steps"], dtype=np.float64)
    values = np.asarray(metrics[metric_name]["values"], dtype=np.float64)
    if len(steps) != len(values):
        raise ValueError(
            f"steps/values length mismatch for '{metric_name}' in {metrics_json_path}: {len(steps)} vs {len(values)}"
        )
    return steps, values


def _plot_one(run_id, metrics_json_path, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, values = _load_metric(metrics_json_path, "test_return_mean")
    x = steps / 1e7

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, values, linewidth=1.5)
    plt.xlabel("t_env (x1e7)")
    plt.ylabel("test_return_mean")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_compare(run_a_id, run_a_path, run_b_id, run_b_path, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_a, values_a = _load_metric(run_a_path, "test_return_mean")
    steps_b, values_b = _load_metric(run_b_path, "test_return_mean")

    x_a = steps_a / 1e7
    x_b = steps_b / 1e7

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_a, values_a, linewidth=1.5, label=f"run{run_a_id}")
    plt.plot(x_b, values_b, linewidth=1.5, label=f"run{run_b_id}")
    plt.xlabel("t_env (x1e7)")
    plt.ylabel("test_return_mean")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    # This script is intended for quick, reproducible plotting from Sacred metrics.
    # Requested: plot run12 and run21 test_return_mean vs t_env (scaled by 1e7).
    base = "/root/research/results/sacred/smpe/Foraging-2s-9x9-3p-2f-coop-v2"
    run12 = os.path.join(base, "12", "metrics.json")
    run21 = os.path.join(base, "21", "metrics.json")

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    out12 = os.path.join(out_dir, "run12_test_return_mean.png")
    out21 = os.path.join(out_dir, "run21_test_return_mean.png")
    out_cmp = os.path.join(out_dir, "run12_vs_run21_test_return_mean.png")

    _plot_one(
        run_id=12,
        metrics_json_path=run12,
        out_path=out12,
        title="run12: test_return_mean vs t_env",
    )
    _plot_one(
        run_id=21,
        metrics_json_path=run21,
        out_path=out21,
        title="run21: test_return_mean vs t_env",
    )
    _plot_compare(
        run_a_id=12,
        run_a_path=run12,
        run_b_id=21,
        run_b_path=run21,
        out_path=out_cmp,
        title="run12 vs run21: test_return_mean vs t_env",
    )

    print("Saved plots:")
    print(out12)
    print(out21)
    print(out_cmp)


if __name__ == "__main__":
    main()

