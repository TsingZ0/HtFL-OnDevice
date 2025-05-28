import argparse
import os
import time
import torch
import flwr as fl
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from collections import defaultdict
from logging import WARNING, INFO
from colext import MonitorFlwrStrategy

from .utils.misc import weighted_metrics_avg


def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters

@MonitorFlwrStrategy
class FD(fl.server.strategy.FedAvg):
    def __init__(self,
            fraction_fit,
            fraction_evaluate,
            min_fit_clients,
            min_available_clients,
            fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn,
            evaluate_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
            inplace,
        ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            inplace=inplace,
        )

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        uploaded_protos_per_client = []
        for _, fit_res in results:
            client_protos = {}
            for label, proto in enumerate(parameters_to_ndarrays(fit_res.parameters)):
                if len(proto.shape) > 0:
                    client_protos[label] = torch.tensor(proto)
            uploaded_protos_per_client.append(client_protos)

        global_protos = proto_cluster(uploaded_protos_per_client)

        global_protos_ndarrays = [0 for _ in range(len(global_protos))]
        for label, proto in global_protos.items():
            print(f"global_protos_ndarrays size= {len(global_protos_ndarrays)}", flush=True)
            print(f"label= {label}", flush=True)
            global_protos_ndarrays[label] = proto.cpu().numpy()
        parameters_aggregated = ndarrays_to_parameters(global_protos_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


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

    # Start server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=FD(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            fit_metrics_aggregation_fn=None,
            evaluate_metrics_aggregation_fn=weighted_metrics_avg,
            evaluate_fn=None,
            on_fit_config_fn=None,
            on_evaluate_config_fn=None,
            inplace=False,
        ),
    )

