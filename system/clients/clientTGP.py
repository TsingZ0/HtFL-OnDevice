import argparse
import copy
import os
import time
import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict, defaultdict
from flwr.common.logger import log
from logging import WARNING, INFO

from .clientBase import ClientBase
from .utils.models import get_model, save_item, load_item


def agg_func(protos):
    for [label, logit_list] in protos.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            protos[label] = logit / len(logit_list)
        else:
            protos[label] = logit_list[0]
    return protos


class Client(ClientBase):
    def __init__(self, args, model):
        super().__init__(args, model)

        self.loss_mse = nn.MSELoss()

    # send
    def get_parameters(self, config):
        self.collect_protos()
        protos = load_item("protos", self.args.save_folder_path)
        uploads = [0 for _ in range(self.args.num_classes)]
        for key, value in protos.items():
            uploads[key] = value.cpu().numpy()
        return uploads

    # receive
    def set_parameters(self, protos):
        proto_dict = zip(range(self.args.num_classes), protos)
        protos = OrderedDict(
            {key: torch.tensor(value).to(self.device) for key, value in proto_dict}
        )
        save_item(protos, "protos", self.args.save_folder_path)

    def train(self):
        """Train the network on the training set."""
        model = load_item("model", self.args.save_folder_path)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum
        )
        protos = load_item("protos", self.args.save_folder_path)
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                reps = model.base(images)
                outputs = model.head(reps)
                loss = criterion(outputs, labels)

                if protos is not None:
                    proto_new = copy.deepcopy(reps.detach())
                    for i, label in enumerate(labels):
                        label = label.item()
                        proto_new[i, :] = protos[label].data
                    loss += self.loss_mse(proto_new, reps) * self.args.lamda

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        save_item(model, "model", self.args.save_folder_path)

    def test(self):
        """Validate the model on the entire test set."""
        model = load_item("model", self.args.save_folder_path)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
        correct, total, loss = 0, 0, 0.0
        protos = load_item("protos", self.args.save_folder_path)
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                reps = model.base(images)
                outputs = model.head(reps)
                loss += (criterion(outputs, labels)).sum().item()
                total += labels.size(0)
                if protos is not None:
                    dists = float('inf') * torch.ones(
                        labels.shape[0], self.args.num_classes).to(self.device)
                    for i, r in enumerate(reps):
                        for j, pro in protos.items():
                            dists[i, j] = self.loss_mse(r, pro)
                    _, predicted = torch.min(dists.data, 1)
                    correct += (predicted == labels).sum().item()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
        loss = loss / total
        accuracy = correct / total
        return loss, accuracy

    def collect_protos(self):
        model = load_item("model", self.args.save_folder_path)
        model.eval()
        protos = defaultdict(list)
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                reps = model.base(images)
                for i, label in enumerate(labels):
                    label = label.item()
                    protos[label].append(reps[i, :].detach().data)
        protos = agg_func(protos)
        save_item(protos, "protos", self.args.save_folder_path)


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
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--lamda", type=float, default=1.0)
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
