
import os
import torch
from flwr.common.logger import log
from logging import WARNING, INFO


def weighted_metrics_avg(eval_metrics):
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in eval_metrics)
    metrics_aggregated = {}
    for num_examples, metrics in eval_metrics:
        for key, value in metrics.items():
            if key not in metrics_aggregated:
                metrics_aggregated[key] = num_examples * value
            else:
                metrics_aggregated[key] += num_examples * value
    for key in metrics_aggregated:
        metrics_aggregated[key] /= num_total_evaluation_examples
        log(INFO, f"{key}: {metrics_aggregated[key]}")
    return metrics_aggregated

def save_item(item, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, f"{item_name}.pt"))

def load_item(item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, f"{item_name}.pt"))
    except FileNotFoundError:
        log(INFO, f"Not Found: {item_name}")
        return None
