import argparse
import os
import time
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from logging import WARNING, INFO
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate
from torch.utils.data import DataLoader
from utils.misc import weighted_metrics_avg, save_item, load_item
from collections import OrderedDict


def get_head(args):
    if 'ResNet' in args.client_model:
        head = nn.Linear(args.feature_dim, args.num_classes)
    else:
        raise NotImplementedError
    return head


# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device) # sampling from Gaussian

        y_input = F.one_hot(labels, self.num_classes)
        z = torch.cat((eps, y_input), dim=1)

        z = self.fc1(z)
        z = self.fc(z)

        return z


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
        generative_model = Generator(
            noise_dim=args.noise_dim, 
            num_classes=args.num_classes, 
            hidden_dim=args.hidden_dim, 
            feature_dim=args.feature_dim, 
            device=self.device
        ).to(self.device)
        save_item(generative_model, 'generative_model', self.args.save_folder_path)

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
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results)

        # Update generator
        uploaded_heads = []
        for weights, num_examples in weights_results:
            head = get_head(self.args).to(self.device)
            state_dict = OrderedDict({key: torch.tensor(value) 
                for key, value in zip(head.state_dict().keys(), weights)})
            head.load_state_dict(state_dict, strict=True)
            uploaded_heads.append((head, num_examples))
        self.update_generative_model(uploaded_heads)
        generative_model = load_item('generative_model', self.args.save_folder_path)
        generative_model_param_ndarrays = [val.cpu().numpy() for _, val in generative_model.state_dict().items()]

        parameters_aggregated = ndarrays_to_parameters(
            aggregated_ndarrays + generative_model_param_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    def update_generative_model(self, uploaded_heads):
        generative_model = load_item('generative_model', self.args.save_folder_path)
        generative_model.train()
        generative_model_opt = torch.optim.Adam(
            params=generative_model.parameters(),
            lr=self.args.learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.CrossEntropyLoss()
        candidate_labels = np.arange(self.args.num_classes)
        for _ in range(self.args.epochs):
            labels = np.random.choice(candidate_labels, self.args.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            z = generative_model(labels)
            logits = 0
            for head, num_examples in uploaded_heads:
                head.eval()
                logits += head(z) * num_examples

            generative_model_opt.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            generative_model_opt.step()
        save_item(generative_model, 'generative_model', self.args.save_folder_path)


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
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
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

