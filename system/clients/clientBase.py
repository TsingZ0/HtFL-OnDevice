import warnings
import torch
import torchvision
import torchvision.transforms as transforms
import flwr as fl
from torch.utils.data import DataLoader
from flwr.common.logger import log
from logging import WARNING, INFO

warnings.simplefilter("ignore")


class ClientBase(fl.client.NumPyClient):
    def __init__(self, args, model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.model = model.to(self.device)
        self.load_data()

    # send
    def get_parameters(self, config):
        raise NotImplementedError

    # receive
    def set_parameters(self, parameters):
        raise NotImplementedError

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        log(INFO, "Before local training\t Loss: {:.4f}, Accuracy: {:.4f}".format(loss, accuracy))
        self.train()
        loss, accuracy = self.test()
        log(INFO, "After local training\t Loss: {:.4f}, Accuracy: {:.4f}".format(loss, accuracy))
        uploads = self.get_parameters(config={})
        num_train_examples = self.num_examples["trainset"]
        metrics = {}
        return uploads, num_train_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        loss = float(loss)
        num_test_examples = self.num_examples["testset"]
        metrics = {
            "accuracy": float(accuracy)
        }
        return loss, num_test_examples, metrics

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
        """Train the model on the training set."""
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self):
        """Validate the model on the entire test set."""
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                loss += (criterion(outputs, labels)).sum().item()
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        loss = loss / total
        accuracy = correct / total
        return loss, accuracy

