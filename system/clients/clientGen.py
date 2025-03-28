import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from collections import OrderedDict
from flwr.common.logger import log
from logging import WARNING, INFO

from .clientBase import ClientBase
from .utils.models import get_model, save_item, load_item


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


class Client(ClientBase):
    def __init__(self, args, model):
        super().__init__(args, model)
        generative_model = Generator(
            noise_dim=args.noise_dim,
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            feature_dim=args.feature_dim,
            device=self.device
        ).to(self.device)
        save_item(generative_model, 'generative_model', self.args.save_folder_path)

    # send
    def get_parameters(self, config):
        model = load_item("model", self.args.save_folder_path)
        return [val.cpu().numpy() for _, val in model.head.state_dict().items()]

    # receive
    def set_parameters(self, parameters):
        model = load_item("model", self.args.save_folder_path)
        params_len = len(model.head.state_dict().keys())
        if len(parameters) != params_len:
            model = load_item("model", self.args.save_folder_path)
            params_dict = zip(model.head.state_dict().keys(), parameters[:params_len])
            state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
            model.head.load_state_dict(state_dict, strict=True)
            save_item(model, "model", self.args.save_folder_path)

            generative_model = load_item("generative_model", self.args.save_folder_path)
            params_dict = zip(generative_model.state_dict().keys(), parameters[params_len:])
            state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
            generative_model.load_state_dict(state_dict, strict=True)
            save_item(generative_model, "generative_model", self.args.save_folder_path)
        else:
            log(WARNING, "Received parameters are only for initialization.")

    def train(self):
        """Train the model on the training set."""
        model = load_item("model", self.args.save_folder_path)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum
        )
        generative_model = load_item("generative_model", self.args.save_folder_path)
        generative_model.eval()
        candidate_labels = np.arange(self.args.num_classes)
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                if generative_model is not None:
                    labels = np.random.choice(candidate_labels, self.args.batch_size)
                    labels = torch.LongTensor(labels).to(self.device)
                    z = generative_model(labels)
                    loss += criterion(model.head(z), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        save_item(model, "model", self.args.save_folder_path)


if __name__ == "__main__":
    # Configuration of the client
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder_path", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()
    timestamp = str(time.time())
    log(INFO, f"Timestamp: {timestamp}")
    args.save_folder_path = os.path.join(args.save_folder_path, timestamp)

    # Load model
    model = get_model(args)

    # Start client
    fl.client.start_client(
        server_address=args.server_address,
        client=Client(args, model).to_client()
    )
