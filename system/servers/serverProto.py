import argparse
import flwr as fl
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


if __name__ == "__main__":
    # Configration of the server
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--min_fit_clients", type=int, default=2)
    parser.add_argument("--min_available_clients", type=int, default=2)
    args = parser.parse_args()

    # Start server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            fit_metrics_aggregation_fn=None,
            evaluate_metrics_aggregation_fn=weighted_metrics_avg,
            evaluate_fn=None,
            on_fit_config_fn=None,
            on_evaluate_config_fn=None,
            inplace=True,
        ),
    )

