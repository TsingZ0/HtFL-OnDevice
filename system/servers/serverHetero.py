import argparse
import os
import copy
import time
import flwr as fl
import logging
from collections import OrderedDict
from logging import WARNING, INFO
from utils.misc import weighted_metrics_avg

import torch
from flwr.common.logger import log
from flwr.common import (
    FitIns,
    EvaluateIns,
    GetPropertiesIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from system.clients.utils.models import get_model
from system.clients.utils.submodels import get_submodel, aggregate_submodels


class HeterogeneousClientManager(SimpleClientManager):

    def __init__(self, n_clients):
        super().__init__()
        self.client_to_model_rate_mapping = {}
        self.n_clients = n_clients
        self.model_config = None

    def unregister(self, client: ClientProxy) -> None:
        super().unregister(client)
        if client.cid in self.client_to_model_rate_mapping:
            del self.client_to_model_rate_mapping[client.cid]

    def sync_client_to_capacity(self, clients):
        # cannot use this in .register because the servicer did not start to listen to client
        # requests yet...
        print("Syncing client to capacity")
        for client_proxy in clients:
            print(client_proxy)
            print("------")
            cid = client_proxy.cid
            noins = GetPropertiesIns({})
            if cid not in self.client_to_model_rate_mapping:
                res = client_proxy.get_properties(noins, None, None)
                print(client_proxy)
                print(type(client_proxy))
                capacity = res.properties.pop("model_rate")
                self.client_to_model_rate_mapping[cid] = capacity

                if self.model_config is None:
                    self.model_config = argparse.Namespace(**res.properties)
                assert self.model_config == argparse.Namespace(**res.properties)
                print(f"{cid} has capacity {capacity}")

    def sample(self, *args, **kwargs):
        clients = super().sample(*args, **kwargs)
        self.sync_client_to_capacity(clients)
        return clients

    def get_client_rate(self, cid):
        return self.client_to_model_rate_mapping[cid]

    def get_model_config(self):
        return self.model_config


class HeteroFL(fl.server.strategy.FedAvg):

    def initialize_parameters(self, client_manager):
        client_manager.wait_for(self.min_available_clients)
        client_manager.sync_client_to_capacity(client_manager.all().values())
        self.previous_model = get_model(client_manager.get_model_config())
        parameters = [val.cpu().numpy() for _, val in self.previous_model.state_dict().items()]
        return ndarrays_to_parameters(parameters)

    def construct_submodels(self, global_parameters, client_manager, config, stage):
        if stage == "train":
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
        elif stage == "evaluate":
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        params_arrays = parameters_to_ndarrays(global_parameters)
        global_model = get_model(client_manager.get_model_config())
        global_model.load_state_dict({
            key: torch.tensor(value)
            for key, value in zip(global_model.state_dict().keys(), params_arrays)
        })
        ins = []
        ins_cls = FitIns if stage == "train" else EvaluateIns
        for client in clients:
            cid = client.cid
            client_rate = client_manager.get_client_rate(cid)
            client_model = get_submodel(copy.deepcopy(global_model), client_rate)
            print("Constructing submodel for client", cid, "with rate", client_rate, "and model size", sum(p.numel() for p in client_model.parameters()))
            client_parameters = [
                val.cpu().numpy() for _, val in client_model.state_dict().items()
            ]
            ins.append((
                client,
                ins_cls(ndarrays_to_parameters(client_parameters), config)
            ))
        return ins

    def set_parameters(self, model, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        self.set_parameters(self.previous_model, parameters_to_ndarrays(parameters))
        self.client_to_rate_mapping = client_manager.client_to_model_rate_mapping
        return self.construct_submodels(parameters, client_manager, config, "train")

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom fit config function provided
            config = self.on_evaluate_config_fn(server_round)
        return self.construct_submodels(parameters, client_manager, config, "evaluate")

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        submodel_list = []
        for proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            model = copy.deepcopy(self.previous_model)
            model = get_submodel(model, self.client_to_rate_mapping[proxy.cid])
            self.set_parameters(model, params)
            submodel_list.append(model)

        aggregated_ndarrays = aggregate_submodels(
            previous_model=self.previous_model,
            submodels_list=submodel_list
        )

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

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
    print("starting...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder_path", type=str, default="checkpoints")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--min_fit_clients", type=int, default=2)
    parser.add_argument("--min_available_clients", type=int, default=2)
    args = parser.parse_args()
    print("parsed...")
    timestamp = str(time.time())
    logging.basicConfig(level=logging.INFO)

    log(INFO, f"Timestamp: {timestamp}")
    args.save_folder_path = os.path.join(args.save_folder_path, timestamp)
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=HeteroFL(
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
        client_manager=HeterogeneousClientManager(n_clients=args.min_available_clients),
    )
