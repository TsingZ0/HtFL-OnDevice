import os
import numpy as np
import torch

DEVICE_DATA_DIR = "device_data"
def read_data(file_path):
    np_data = np.load(file_path, allow_pickle=True)['data'].tolist()

    X = torch.Tensor(np_data['x']).type(torch.float32)
    Y = torch.Tensor(np_data['y']).type(torch.int64)
    torch_data = [(x, y) for x, y in zip(X, Y)]

    return torch_data

def read_client_local_data(train=True):
    # Read local data relative to the current working directory - should be root of project
    client_id = os.environ.get('COLEXT_CLIENT_ID')
    if not client_id:
        raise ValueError("COLEXT_CLIENT_ID env variable is not defined. \
                         Are you using a CoLExT env?")
    data_type = 'train' if train else 'test'
    file_path = os.path.join(DEVICE_DATA_DIR, f"{data_type}_{client_id}.npz")

    data = read_data(file_path)
    return data

def has_local_device_data():
    return os.path.exists(DEVICE_DATA_DIR)
