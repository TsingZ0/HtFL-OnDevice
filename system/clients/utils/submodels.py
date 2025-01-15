import numpy as np
import torch
import torch.nn as nn

from system.clients.utils.models import HARCNN, BaseHeadSplit, SimpleCNN


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        assert 0 < rate <= 1.
        self.rate = rate

    def forward(self, inp):
        return inp / self.rate if self.train else inp


def _submodel_linear_layer(layer, in_features, out_features, rate):
    layer.weight.data = layer.weight.data[:out_features, :in_features]
    layer.bias.data = layer.bias.data[:out_features]
    layer.in_features = in_features
    layer.out_features = out_features
    return nn.Sequential(
        layer,
        Scaler(rate)
    )


def _submodel_conv_layer(layer, in_channels, out_channels, rate):
    conv_config = [torch.arange(out_channels), torch.arange(in_channels)]
    layer.in_channels = in_channels
    layer.out_channels = out_channels
    layer.weight.data = layer.weight.data[torch.meshgrid(conv_config, indexing="ij")]
    layer.bias.data = layer.bias.data[conv_config[0]]
    return nn.Sequential(
        layer,
        Scaler(rate)
    )


def _get_submodel_HARCNN(model, rate):
    conv1 = model.conv1
    conv1_out_channels = int(conv1[0].out_channels * rate)

    conv1[0] = _submodel_conv_layer(conv1[0], conv1[0].in_channels, conv1_out_channels, rate)

    conv2 = model.conv2
    conv2_in_channels = conv1_out_channels
    conv2_out_channels = int(conv2[0].out_channels * rate)
    conv2[0] = _submodel_conv_layer(conv2[0], conv2_in_channels, conv2_out_channels, rate)

    fc_sizes = [
        int(64 * 26 * rate),
        int(model.fc[0].out_features * rate),
        int(model.fc[2].out_features * rate),
        model.fc[-1].out_features,  # do not reduce the number of output features (i.e., classes)
    ]

    model.fc[0] = _submodel_linear_layer(model.fc[0], fc_sizes[0], fc_sizes[1], rate)
    model.fc[2] = _submodel_linear_layer(model.fc[2], fc_sizes[1], fc_sizes[2], rate)
    model.fc[4] = _submodel_linear_layer(model.fc[4], fc_sizes[2], fc_sizes[3], rate)


def _get_submodel_SimpleCNN(model, rate):
    conv1 = model.base.encoder[0]
    conv1_out_channels = int(conv1.out_channels * rate)

    model.base.encoder[0] = _submodel_conv_layer(conv1, conv1.in_channels, conv1_out_channels, rate)

    conv2 = model.base.encoder[3]
    conv2_in_channels = conv1_out_channels
    conv2_out_channels = int(conv2.out_channels * rate)
    model.base.encoder[3] = _submodel_conv_layer(conv2, conv2_in_channels, conv2_out_channels, rate)
    feature_size = int(model.base.fc.output_size * rate)
    model.base.fc = nn.AdaptiveAvgPool1d(feature_size)
    model.head = _submodel_linear_layer(model.head, feature_size, model.head.out_features, rate)
    return model


def get_submodel(model, rate):
    print(isinstance(model, BaseHeadSplit))
    print(isinstance(model.base, SimpleCNN))
    if isinstance(model, HARCNN):
        return _get_submodel_HARCNN(model, rate)
    if isinstance(model, BaseHeadSplit) and isinstance(model.base, SimpleCNN):
        return _get_submodel_SimpleCNN(model, rate)
    raise NotImplementedError("Submodel construction not implemented for {}", model.__class__.__name__)


def aggregate_submodels(
    previous_model,
    submodels_list,
):
    layers = dict(previous_model.named_modules())
    aggregate_layer_names = [k for k, v in layers.items() if isinstance(v, (nn.Linear, nn.Conv2d))]

    out_params = []
    for layer_name in aggregate_layer_names:
        layer = layers[layer_name]

        # weight/bias sum/count
        ws, bs = torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
        wc, bc = torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)

        for submodel in submodels_list:
            # add a 0 because for every layer in the input model that gets
            # reduced we introduce a Sequential layer with a Scaler layer
            submodel_weight = submodel.state_dict()[f"{layer_name}.0.weight"]
            submodel_bias = submodel.state_dict()[f"{layer_name}.0.bias"]

            submodel_layer_conf = [
                torch.arange(submodel_weight.shape[0]),
                torch.arange(submodel_weight.shape[1])
            ]

            idxs = torch.meshgrid(submodel_layer_conf, indexing="ij")
            ws[idxs] += submodel_weight
            wc[idxs] += 1

            bs[submodel_layer_conf[0]] += submodel_bias
            bc[submodel_layer_conf[0]] += 1
        out_params.append(torch.where(wc > 0, ws / wc, layer.weight))
        out_params.append(torch.where(bc > 0, bs / bc, layer.bias))
    return [p.detach().numpy() for p in out_params]
