import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import flwr as fl
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# Define the local model
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x

class Client(fl.client.NumPyClient):
    def __init__(self, args, net):
        self.args = args
        self.net = net
        self.load_data()

    def get_parameters(self, config):
        head = self.net.fc
        return [val.cpu().numpy() for _, val in head.state_dict().items()]

    def set_parameters(self, parameters):
        head = self.net.fc
        params_dict = zip(head.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        head.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    # rewite this code to use already assigned local data
    def load_data(self):
        """Load training and test set."""
        trainset = torchvision.datasets.CIFAR10(
            "test_data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10("test_data", train=False, download=True, transform=transform)

        self.trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.args.batch_size)
        self.num_examples = {"trainset" : len(trainset), "testset" : len(testset)}

    def train(self):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.learning_rate)
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()

    def test(self):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy


if __name__ == "__main__":
    # Configration of the client
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    # Load model and data
    # net = torchvision.models.resnet18(pretrained=False, num_classes=10).to(DEVICE)
    net = Net().to(DEVICE)

    # Start client
    fl.client.start_client(
        server_address=args.server_address, 
        client=Client(args, net).to_client()
    )

