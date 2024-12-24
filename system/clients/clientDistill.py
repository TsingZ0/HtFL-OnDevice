import argparse
import copy
import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict, defaultdict
from clientBase import ClientBase
from utils.models import get_model


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
        self.protos = None

    # send
    def get_parameters(self, config):
        self.collect_protos()
        uploads = [0 for _ in range(self.args.num_classes)]
        for key, value in self.protos.items():
            uploads[key] = value.cpu().numpy()
        return uploads

    # receive
    def set_parameters(self, protos):
        proto_dict = zip(range(self.args.num_classes), protos)
        self.protos = OrderedDict({key: torch.tensor(value) for key, value in proto_dict})

    def train(self):
        """Train the network on the training set."""
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.args.learning_rate, 
            momentum=self.args.momentum
        )
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                if self.protos is not None:
                    proto_new = copy.deepcopy(outputs.detach())
                    for i, label in enumerate(labels):
                        label = label.item()
                        proto_new[i, :] = self.protos[label].data
                    loss += self.loss_mse(proto_new, outputs) * self.args.lamda

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def collect_protos(self):
        self.model.eval()
        protos = defaultdict(list)
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                for i, label in enumerate(labels):
                    label = label.item()
                    protos[label].append(outputs[i, :].detach().data)
        self.protos = agg_func(protos)


if __name__ == "__main__":
    # Configuration of the client
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--lamda", type=float, default=1.0)
    args = parser.parse_args()

    # Load model
    model = get_model(args)

    # Start client
    fl.client.start_client(
        server_address=args.server_address, 
        client=Client(args, model).to_client()
    )
