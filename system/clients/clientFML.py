import argparse
import os
import time
import torch
import torch.nn as nn
import flwr as fl
import torch.nn.functional as F
from collections import OrderedDict
from flwr.common.logger import log
from logging import WARNING, INFO
from colext import MonitorFlwrClient

from .clientBase import ClientBase
from .utils.models import get_model, get_auxiliary_model, save_item, load_item

@MonitorFlwrClient
class Client(ClientBase):
    def __init__(self, args, model, auxiliary_model):
        super().__init__(args, model)
        save_item(auxiliary_model.to(self.device), "auxiliary_model", self.args.save_folder_path)
        self.KL = nn.KLDivLoss()

    # send
    def get_parameters(self, config):
        auxiliary_model = load_item("auxiliary_model", self.args.save_folder_path)
        return [val.cpu().numpy() for _, val in auxiliary_model.state_dict().items()]

    # receive
    def set_parameters(self, parameters):
        auxiliary_model = load_item("auxiliary_model", self.args.save_folder_path)
        params_dict = zip(auxiliary_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
        auxiliary_model.load_state_dict(state_dict, strict=True)
        save_item(auxiliary_model, "auxiliary_model", self.args.save_folder_path)

    def train(self):
        """Train the model on the training set."""
        model = load_item("model", self.args.save_folder_path)
        model.train()
        auxiliary_model = load_item("auxiliary_model", self.args.save_folder_path)
        auxiliary_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum
        )
        optimizer_aux = torch.optim.SGD(
            auxiliary_model.parameters(),
            lr=self.args.auxiliary_learning_rate,
            momentum=self.args.momentum
        )
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                outputs_aux = auxiliary_model(images)
                loss = criterion(outputs, labels) * self.args.alpha + self.KL(F.log_softmax(outputs, dim=1), F.softmax(outputs_aux, dim=1)) * (1-self.args.alpha)
                loss_aux = criterion(outputs_aux, labels) * self.args.beta + self.KL(F.log_softmax(outputs_aux, dim=1), F.softmax(outputs, dim=1)) * (1-self.args.beta)
                optimizer.zero_grad()
                optimizer_aux.zero_grad()
                loss.backward(retain_graph=True)
                loss_aux.backward(retain_graph=True)
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(auxiliary_model.parameters(), 10)
                optimizer.step()
                optimizer_aux.step()
        save_item(model, "model", self.args.save_folder_path)
        save_item(auxiliary_model, "auxiliary_model", self.args.save_folder_path)


if __name__ == "__main__":
    # Configuration of the client
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder_path", type=str, default="checkpoints")
    parser.add_argument("--data_name", type=str, default="iWildCam")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--auxiliary_learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--auxiliary_model", type=str, default="ResNet4")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()
    timestamp = str(time.time())
    log(INFO, f"Timestamp: {timestamp}")
    args.save_folder_path = os.path.join(args.save_folder_path, timestamp)

    # Load model
    model = get_model(args)
    auxiliary_model = get_auxiliary_model(args)

    # Start client
    fl.client.start_client(
        server_address=args.server_address,
        client=Client(args, model, auxiliary_model).to_client()
    )
