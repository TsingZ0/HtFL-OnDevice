import argparse
import torch
import flwr as fl
from collections import OrderedDict
from clientBase import ClientBase
from utils.models import get_model


class Client(ClientBase):
    def __init__(self, args, net):
        super().__init__(args, net)

    def get_parameters(self, config):
        head = self.net.fc
        return [val.cpu().numpy() for _, val in head.state_dict().items()]

    def set_parameters(self, parameters):
        head = self.net.fc
        params_dict = zip(head.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        head.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    # Configuration of the client
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    # Load model
    net = get_model(args)

    # Start client
    fl.client.start_client(
        server_address=args.server_address, 
        client=Client(args, net).to_client()
    )
