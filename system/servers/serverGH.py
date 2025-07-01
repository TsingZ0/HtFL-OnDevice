import argparse
import os
import time
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from logging import WARNING, INFO
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from torch.utils.data import DataLoader
from colext import MonitorFlwrStrategy

from .utils.misc import weighted_metrics_avg, save_item, load_item


def get_head(args):
    if 'ResNet' in args.client_model:
        head = nn.Linear(args.feature_dim, args.num_classes)
    else:
        raise NotImplementedError
    return head

@MonitorFlwrStrategy
class FedGH(fl.server.strategy.FedAvg):
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
            args,
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
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Head = get_head(args).to(self.device)
        save_item(Head, 'Head', self.args.save_folder_path)

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
        uploaded_protos = []
        for _, fit_res in results:
            client_protos = {}
            for label, proto in enumerate(parameters_to_ndarrays(fit_res.parameters)):
                if len(proto.shape) > 0:
                    client_protos[label] = torch.tensor(proto)
                    uploaded_protos.append((torch.tensor(proto), label))

        # Update Head
        self.update_Head(uploaded_protos)
        Head = load_item('Head', self.args.save_folder_path)
        Head_ndarrays = [val.cpu().numpy() for _, val in Head.state_dict().items()]

        parameters_aggregated = ndarrays_to_parameters(Head_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def update_Head(self, uploaded_protos):
        Head = load_item('Head', self.args.save_folder_path)
        Head.train()
        Head_opt = torch.optim.SGD(Head.parameters(), lr=self.args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        proto_loader = DataLoader(
            uploaded_protos,
            self.args.batch_size,
            drop_last=False,
            shuffle=True
        )
        for _ in range(self.args.epochs):
            for proto, y in proto_loader:
                proto = proto.to(self.device)
                y = torch.Tensor(y).type(torch.int64).to(self.device)
                out = Head(proto)
                loss = criterion(out, y)
                Head_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(Head.parameters(), 10)
                Head_opt.step()
        save_item(Head, 'Head', self.args.save_folder_path)


if __name__ == "__main__":
    # Configration of the server
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder_path", type=str, default="checkpoints")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--min_fit_clients", type=int, default=2)
    parser.add_argument("--min_available_clients", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--client_model", type=str, default="ResNet")
    args = parser.parse_args()
    timestamp = str(time.time())
    log(INFO, f"Timestamp: {timestamp}")
    args.save_folder_path = os.path.join(args.save_folder_path, timestamp)

    # Start server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=FedGH(
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
            args=args,
        ),
    )

