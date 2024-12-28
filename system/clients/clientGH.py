import argparse
import os
import time
import torch
import flwr as fl
from collections import OrderedDict, defaultdict
from clientBase import ClientBase
from utils.models import get_model, save_item, load_item
from flwr.common.logger import log
from logging import WARNING, INFO


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

    # send
    def get_parameters(self, config):
        self.collect_protos()
        protos = load_item("protos", self.args.save_folder_path)
        protos_upload = [0 for _ in range(self.args.num_classes)]
        for key, value in protos.items():
            protos_upload[key] = value.cpu().numpy()
        return protos_upload

    # receive
    def set_parameters(self, parameters):
        if parameters[0].shape[0] == self.args.feature_dim:
            log(WARNING, "Received parameters are only for initialization.")
        else:
            model = load_item("model", self.args.save_folder_path)
            params_dict = zip(model.head.state_dict().keys(), parameters)
            state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
            model.head.load_state_dict(state_dict, strict=True)
            save_item(model, "model", self.args.save_folder_path)

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
    parser.add_argument("--save_folder_path", type=str, default='checkpoints')
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

    # Start client
    fl.client.start_client(
        server_address=args.server_address, 
        client=Client(args, model).to_client()
    )
