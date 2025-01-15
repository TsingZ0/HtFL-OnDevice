import argparse
import os
import time
import torch
import flwr as fl
from collections import OrderedDict
from clientBase import ClientBase
from system.clients.utils.models import get_model, save_item, load_item
from flwr.common.logger import log
from logging import WARNING, INFO

from utils.submodels import get_submodel


class Client(ClientBase):
    def __init__(self, args, model):
        super().__init__(args, model)
        print("initialized client")
        self.model_name = args.model
        self.feature_dim = args.feature_dim
        self.num_classes = args.num_classes

    def get_parameters(self, config):
        model = load_item("model", self.args.save_folder_path)
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        model = load_item("model", self.args.save_folder_path)
        print("Training...", sum(p.numel() for p in model.parameters()))
        del model
        tmp = super().fit(parameters, config)
        print("number of client parameters", sum(t.size for t in tmp[0]))
        return tmp

    # receive
    def set_parameters(self, parameters):
        model = load_item("model", self.args.save_folder_path)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
        model.load_state_dict(state_dict, strict=True)
        save_item(model, "model", self.args.save_folder_path)

    def get_properties(self, config):
        return {
            "model_rate": self.args.rate,
            "model": self.model_name,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
        }


if __name__ == "__main__":
    # Configuration of the client
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder_path", type=str, default="checkpoints")
    parser.add_argument("--rate", type=float, required=True,
                        help="Relative size of the submodel")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()
    timestamp = str(time.time())

    log(INFO, f"Timestamp: {timestamp}")
    args.save_folder_path = os.path.join(args.save_folder_path, timestamp)

    # Load model
    model = get_model(args)
    model = get_submodel(model, args.rate)
    # Start client
    fl.client.start_client(
        server_address=args.server_address,
        client=Client(args, model).to_client()
    )
