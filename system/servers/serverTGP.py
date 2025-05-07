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
from collections import defaultdict
from torch.utils.data import DataLoader
from colext import MonitorFlwrStrategy

from .utils.misc import weighted_metrics_avg, save_item, load_item


def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters


class Trainable_Global_Prototypes(nn.Module):
    def __init__(self, num_classes, hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out

@MonitorFlwrStrategy
class FedTGP(fl.server.strategy.FedAvg):
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
        TGP = Trainable_Global_Prototypes(
            self.args.num_classes,
            self.args.hidden_dim,
            self.args.feature_dim,
            self.device
        ).to(self.device)
        save_item(TGP, 'TGP', self.args.save_folder_path)

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
        uploaded_protos_per_client = []
        for _, fit_res in results:
            client_protos = {}
            for label, proto in enumerate(parameters_to_ndarrays(fit_res.parameters)):
                if len(proto.shape) > 0:
                    client_protos[label] = torch.tensor(proto)
                    uploaded_protos.append((torch.tensor(proto), label))
            uploaded_protos_per_client.append(client_protos)

        # Calculate class-wise minimum distance
        gap = torch.ones(self.args.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    gap[k1] = torch.min(gap[k1], dis)
                    gap[k2] = torch.min(gap[k2], dis)
        self.min_gap = torch.min(gap)
        for i in range(len(gap)):
            if gap[i] > torch.tensor(1e8, device=self.device):
                gap[i] = self.min_gap
        self.max_gap = torch.max(gap)
        log(INFO, f'class-wise minimum distance {gap}')
        log(INFO, f'min_gap {self.min_gap}')
        log(INFO, f'max_gap {self.max_gap}')

        # Update global prototypes
        global_protos = self.update_TGP(uploaded_protos)

        global_protos_ndarrays = [proto.cpu().numpy() for proto in global_protos]
        parameters_aggregated = ndarrays_to_parameters(global_protos_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def update_TGP(self, uploaded_protos):
        TGP = load_item('TGP', self.args.save_folder_path)
        TGP.train()
        TGP_opt = torch.optim.SGD(TGP.parameters(), lr=self.args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.args.epochs):
            proto_loader = DataLoader(
                uploaded_protos,
                self.args.batch_size,
                drop_last=False,
                shuffle=True
            )
            for proto, y in proto_loader:
                proto = proto.to(self.device)
                y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = TGP(list(range(self.args.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)

                one_hot = F.one_hot(y, self.args.num_classes).to(self.device)
                margin = min(self.max_gap.item(), self.args.margin_threthold)
                dist = dist + one_hot * margin
                loss = criterion(-dist, y)

                TGP_opt.zero_grad()
                loss.backward()
                TGP_opt.step()

        TGP.eval()
        global_protos = [TGP(i).detach() for i in range(self.args.num_classes)]
        save_item(TGP, 'TGP', self.args.save_folder_path)
        return global_protos


if __name__ == "__main__":
    # Configration of the server
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder_path", type=str, default="checkpoints")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--min_fit_clients", type=int, default=2)
    parser.add_argument("--min_available_clients", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--margin_threthold", type=float, default=100.0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    timestamp = str(time.time())
    log(INFO, f"Timestamp: {timestamp}")
    args.save_folder_path = os.path.join(args.save_folder_path, timestamp)

    MonitoredStrategy = MonitorFlwrStrategy(FedTGP)

    # Start server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=MonitoredStrategy(
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

