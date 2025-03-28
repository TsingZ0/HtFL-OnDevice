import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
import torch.nn.functional as F
from collections import OrderedDict
from flwr.common.logger import log
from logging import WARNING, INFO

from .clientBase import ClientBase
from .utils.models import get_model, save_item, load_item


class Client(ClientBase):
    def __init__(self, args, model, auxiliary_model):
        super().__init__(args, model)
        save_item(auxiliary_model.to(self.device), "auxiliary_model", self.args.save_folder_path)
        W_h = nn.Linear(args.global_feature_dim, args.feature_dim, bias=False)
        save_item(W_h.to(self.device), "W_h", self.args.save_folder_path)
        self.energy = args.T_start

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
        W_h = load_item("W_h", self.args.save_folder_path)
        W_h.train()
        auxiliary_model = load_item("auxiliary_model", self.args.save_folder_path)
        auxiliary_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        KL = nn.KLDivLoss()
        MSE = nn.MSELoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum
        )
        optimizer_W = torch.optim.SGD(
            W_h.parameters(),
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
                reps = model.base(images)
                reps_aux = auxiliary_model.base(images)
                outputs = model.head(reps)
                outputs_aux = auxiliary_model.head(reps_aux)

                CE_loss = criterion(outputs, labels)
                CE_loss_aux = criterion(outputs_aux, labels)
                L_d = KL(F.log_softmax(outputs, dim=1),
                         F.softmax(outputs_aux, dim=1)) / (CE_loss + CE_loss_aux)
                L_d_aux = KL(F.log_softmax(outputs_aux, dim=1),
                             F.softmax(outputs, dim=1)) / (CE_loss + CE_loss_aux)
                L_h = MSE(reps, W_h(reps_aux)) / (CE_loss + CE_loss_aux)
                L_h_aux = MSE(reps, W_h(reps_aux)) / (CE_loss + CE_loss_aux)

                loss = CE_loss + L_d + L_h
                loss_aux = CE_loss_aux + L_d_aux + L_h_aux

                optimizer.zero_grad()
                optimizer_aux.zero_grad()
                optimizer_W.zero_grad()
                loss.backward(retain_graph=True)
                loss_aux.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(auxiliary_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(W_h.parameters(), 10)
                optimizer.step()
                optimizer_aux.step()
                optimizer_W.step()
        save_item(model, "model", self.args.save_folder_path)
        save_item(auxiliary_model, "auxiliary_model", self.args.save_folder_path)


if __name__ == "__main__":
    # Configuration of the client
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder_path", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--auxiliary_learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--global_feature_dim", type=int, default=512)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--auxiliary_model", type=str, default="ResNet18")
    parser.add_argument("--T_start", type=float, default=0.95)
    parser.add_argument("--T_end", type=float, default=0.98)
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
