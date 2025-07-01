import os
import random
import time
import warnings
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import flwr as fl
from torch.utils.data import DataLoader
from flwr.common.logger import log
from logging import WARNING, INFO

from .utils.data_utils import read_client_local_data, has_local_device_data
from .utils.models import save_item, load_item

warnings.simplefilter("ignore")

class ClientBase(fl.client.NumPyClient):
    def __init__(self, args, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        save_item(model.to(self.device), "model", self.args.save_folder_path)
        self.remapping_data_labels()
        self.load_data()

    # send
    def get_parameters(self, config):
        raise NotImplementedError

    # receive
    def set_parameters(self, parameters):
        raise NotImplementedError

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # loss, accuracy = self.test()
        # log(INFO, "Before local training\t Loss: {:.4f}, Accuracy: {:.4f}".format(loss, accuracy))
        self.train()
        # loss, accuracy = self.test()
        # log(INFO, "After local training\t Loss: {:.4f}, Accuracy: {:.4f}".format(loss, accuracy))
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
    
    def remapping_data_labels(self):
        data_dir = 'dataset/' + self.args.data_name

        if not os.path.exists(data_dir + "_trim"):
            n_clients = int(os.environ.get("COLEXT_N_CLIENTS"))
            file_ids = [str(a) for a in list(range(n_clients))]

            remap = {}
            label_cnt = 0
            for file_id in file_ids:
                train_file_path = data_dir + "/train/" + file_id + ".npz"
                test_file_path = data_dir + "/test/" + file_id + ".npz"
                train_data = np.load(train_file_path, allow_pickle=True)['data'].tolist()
                test_data = np.load(test_file_path, allow_pickle=True)['data'].tolist()
                for original_label in train_data['y'] + test_data['y']:
                    if original_label not in remap.keys():
                        remap[original_label] = label_cnt
                        label_cnt += 1

            log(INFO, f"Label re-mapping: {remap}")
            log(INFO, f"Total labels: {label_cnt}")

            for file_id in file_ids:
                train_file_path = data_dir + "/train/" + file_id + ".npz"
                test_file_path = data_dir + "/test/" + file_id + ".npz"
                train_data = np.load(train_file_path, allow_pickle=True)['data'].tolist()
                test_data = np.load(test_file_path, allow_pickle=True)['data'].tolist()

                new_labels = []
                for label in train_data['y']:
                    new_labels.append(remap[label])
                train_data['y'] = new_labels
                data_dir_new = data_dir + "_trim" + "/train/"
                os.makedirs(data_dir_new, exist_ok=True)
                train_file_path_new = data_dir_new + file_id + ".npz"
                with open(train_file_path_new, 'wb') as f:
                    np.savez_compressed(f, data=train_data)
                    
                new_labels = []
                for label in test_data['y']:
                    new_labels.append(remap[label])
                test_data['y'] = new_labels
                data_dir_new = data_dir + "_trim" + "/test/"
                os.makedirs(data_dir_new, exist_ok=True)
                test_file_path_new = data_dir_new + file_id + ".npz"
                with open(test_file_path_new, 'wb') as f:
                    np.savez_compressed(f, data=test_data)
        else:
            log(INFO, "Data labels already remapped, skipping...")

    # rewite this code to use already assigned local data
    def load_data(self):
        """Load training and test set."""

        if has_local_device_data():
            log(INFO, "Found device local data")
            trainset = read_client_local_data(train=True)
            testset  = read_client_local_data(train=False)
        # otherwise default to CIFAR10 for debugging
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
            trainset = torchvision.datasets.CIFAR10(
                "test_data", train=True, download=True, transform=transform)
            trainset.data = trainset.data[:1000]
            trainset.targets = trainset.targets[:1000]
            testset = torchvision.datasets.CIFAR10("test_data", train=False, download=True, transform=transform)

        self.trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        self.testloader = DataLoader(testset, batch_size=self.args.batch_size)
        self.num_examples = {"trainset" : len(trainset), "testset" : len(testset)}

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
        for _ in range(self.args.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
        save_item(model, "model", self.args.save_folder_path)

    def test(self):
        """Validate the model on the entire test set."""
        model = load_item("model", self.args.save_folder_path)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(images)
                loss += (criterion(outputs, labels)).sum().item()
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        loss = loss / total
        accuracy = correct / total
        return loss, accuracy
