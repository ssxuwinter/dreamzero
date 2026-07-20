#!/usr/bin/env python3
"""Aggregate the three fixed-boundary quick checks without tuning per episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPISODES = (1925, 16001, 50458)
FEATURE_ORDER = (
    "joint_step",
    "joint_step_change",
    "joint_jerk",
    "direction_change",
    "gripper_value",
    "gripper_change",
)
GROUPS = {
    "joint_step_only": ("joint_step",),
    "joint_only": (
        "joint_step",
        "joint_step_change",
        "joint_jerk",
        "direction_change",
    ),
    "gripper_only": ("gripper_value", "gripper_change"),
    "joint_gripper": FEATURE_ORDER,
    "time_only": ("seconds",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="research/action_phase_segmentation/pilot"
    )
    parser.add_argument(
        "--output-dir",
        default="research/action_phase_segmentation/pilot/user_boundary_summary",
    )
    return parser.parse_args()


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    steps = []
    for episode in EPISODES:
        folder = root / f"user_boundary_{episode}"
        episode_metrics = pd.read_csv(folder / "metrics.csv")
        episode_metrics.insert(0, "episode", episode)
        metrics.append(episode_metrics)

        episode_steps = pd.read_csv(folder / "per_step_features.csv")
        episode_steps.insert(0, "episode", episode)
        steps.append(episode_steps)
    return pd.concat(metrics, ignore_index=True), pd.concat(steps, ignore_index=True)


def loeo_metrics(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = steps[(steps["source"] == "wam") & steps["used"]].copy()
    data["target"] = (data["label"] == "fine").astype(int)
    fold_rows = []
    prediction_rows = []
    for group, columns in GROUPS.items():
        for held_out in EPISODES:
            train = data[data["episode"] != held_out]
            test = data[data["episode"] == held_out]
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=20260719,
                        ),
                    ),
                ]
            )
            model.fit(train[list(columns)], train["target"])
            probability = model.predict_proba(test[list(columns)])[:, 1]
            predicted = (probability >= 0.5).astype(int)
            fold_rows.append(
                {
                    "group": group,
                    "held_out_episode": held_out,
                    "auroc": roc_auc_score(test["target"], probability),
                    "auprc": average_precision_score(test["target"], probability),
                    "balanced_accuracy": balanced_accuracy_score(
                        test["target"], predicted
                    ),
                    "macro_f1": f1_score(test["target"], predicted, average="macro"),
                }
            )
            prediction_rows.extend(
                {
                    "group": group,
                    "episode": held_out,
                    "frame": int(frame),
                    "target": int(target),
                    "probability": float(prob),
                    "prediction": int(pred),
                }
                for frame, target, prob, pred in zip(
                    test["frame"], test["target"], probability, predicted
                )
            )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    overall_rows = []
    for group, group_predictions in predictions.groupby("group"):
        overall_rows.append(
            {
                "group": group,
                "auroc": roc_auc_score(
                    group_predictions["target"], group_predictions["probability"]
                ),
                "auprc": average_precision_score(
                    group_predictions["target"], group_predictions["probability"]
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    group_predictions["target"], group_predictions["prediction"]
                ),
                "macro_f1": f1_score(
                    group_predictions["target"],
                    group_predictions["prediction"],
                    average="macro",
                ),
            }
        )
    return folds, pd.DataFrame(overall_rows)


def plot_summary(
    metrics: pd.DataFrame, overall: pd.DataFrame, output: Path
) -> None:
    wam = metrics[metrics["source"] == "wam"].copy()
    heatmap = wam.pivot(
        index="episode", columns="feature", values="separation_auc"
    ).loc[list(EPISODES), list(FEATURE_ORDER)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    image = axes[0].imshow(heatmap.to_numpy(), vmin=0.5, vmax=1.0, cmap="YlGnBu")
    axes[0].set_xticks(range(len(FEATURE_ORDER)), FEATURE_ORDER, rotation=35, ha="right")
    axes[0].set_yticks(range(len(EPISODES)), [str(value) for value in EPISODES])
    axes[0].set_title("Per-episode WAM feature separation AUROC")
    for row in range(len(EPISODES)):
        for column in range(len(FEATURE_ORDER)):
            value = heatmap.iloc[row, column]
            axes[0].text(column, row, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    ordered_groups = list(GROUPS)
    chart = overall.set_index("group").loc[ordered_groups]
    x = np.arange(len(ordered_groups))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        chart["balanced_accuracy"],
        width,
        label="balanced accuracy",
        color="#2563eb",
    )
    axes[1].bar(
        x + width / 2,
        chart["macro_f1"],
        width,
        label="macro F1",
        color="#f97316",
    )
    axes[1].axhline(0.5, color="#64748b", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, ordered_groups, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Leave-one-episode-out linear classification")
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Fixed labels for all episodes: free <3s, exclude 3-4s, fine >=4s")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, steps = load_inputs(Path(args.root))
    folds, overall = loeo_metrics(steps)

    metrics.to_csv(output_dir / "all_feature_metrics.csv", index=False)
    folds.to_csv(output_dir / "loeo_fold_metrics.csv", index=False)
    overall.to_csv(output_dir / "loeo_overall_metrics.csv", index=False)
    plot_summary(metrics, overall, output_dir / "summary.png")

    wam = metrics[metrics["source"] == "wam"]
    print("WAM per-episode feature metrics")
    print(
        wam[
            [
                "episode",
                "feature",
                "free_mean",
                "fine_mean",
                "separation_auc",
                "fine_direction",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nLeave-one-episode-out overall")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
