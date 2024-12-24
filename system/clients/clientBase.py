import torch
import torchvision
import torchvision.transforms as transforms
import flwr as fl
from torch.utils.data import DataLoader


class ClientBase(fl.client.NumPyClient):
    def __init__(self, args, net):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.net = net.to(self.device)
        self.load_data()

    def get_parameters(self, config):
        raise NotImplementedError

    def set_parameters(self, parameters):
        raise NotImplementedError

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
        transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            "test_data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10("test_data", train=False, download=True, transform=transform)

        self.trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.args.batch_size)
        self.num_examples = {"trainset" : len(trainset), "testset" : len(testset)}

    def train(self):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), 
            lr=self.args.learning_rate, 
            momentum=self.args.momentum
        )
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()

    def test(self):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                loss += (criterion(outputs, labels)).sum().item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss = loss / total
        accuracy = correct / total
        return loss, accuracy

