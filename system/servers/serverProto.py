import argparse
import os
import time
from colext import MonitorFlwrStrategy
import flwr as fl
from flwr.common.logger import log
from logging import WARNING, INFO
from .utils.misc import weighted_metrics_avg


if __name__ == "__main__":
    # Configration of the server
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder_path", type=str, default="checkpoints")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--min_fit_clients", type=int, default=2)
    parser.add_argument("--min_available_clients", type=int, default=2)
    args = parser.parse_args()
    timestamp = str(time.time())
    log(INFO, f"Timestamp: {timestamp}")
    args.save_folder_path = os.path.join(args.save_folder_path, timestamp)

    strategy = MonitorFlwrStrategy(fl.server.strategy.FedAvg)

    # Start server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy(
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

