import copy
import os
import shutil
import yaml

template_experiment = {
    "name": "benchmark_resnet_LG-FedAvg",
    "project": "htfl-ondevice",
    "code": {
        "path": "../../../../../", # root folder
        "client": {
            "command": (
                "./config_device_data.sh ${COLEXT_DATASETS}/iWildCam ${COLEXT_CLIENT_ID} && "
                "python3 -m system.clients.clientLG "
                "--server_address=${COLEXT_SERVER_ADDRESS} "
                "--num_classes=158"
            )
        },
        "server": {
            "command": (
                "python3 -m system.servers.serverLG "
                "--min_fit_clients=${COLEXT_N_CLIENTS} "
                "--min_available_clients=${COLEXT_N_CLIENTS} "
                "--num_rounds=10"
            )
        },
    },
    "clients": [
        {"dev_type": "JetsonAGXOrin",    "count": 2},
        {"dev_type": "JetsonOrinNano",   "count": 2},
        {"dev_type": "JetsonXavierNX",   "count": 2},
        {"dev_type": "JetsonNano",       "count": 2},
        {"dev_type": "OrangePi5B",       "count": 2},
        # {"dev_type": "LattePandaDelta3", "count": 2}, # Facing power measurement issues
    ],
}

models = [
    "ResNet4", "ResNet6", "ResNet8", "ResNet10",
    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"
]
# JetsonNano does not have enough memory for some models
exclude_models_for_dev = {
    "JetsonNano": ["ResNet101", "ResNet152"],
}

# Clean existing output dir before we create new configs
script_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.join(script_dir, "output", "colext_configs")
if os.path.exists(config_dir):
    shutil.rmtree(config_dir)
os.makedirs(config_dir)

# Generate YAML files for each model
for config_id, model_name in enumerate(models):
    filename = os.path.join(config_dir, f"{config_id}_bench_model_{model_name}.yaml")

    exp = copy.deepcopy(template_experiment)
    exp["code"]["client"]["command"] += f" --model={model_name}"

    exp["clients"] = [
        client for client in exp["clients"]
        if model_name not in exclude_models_for_dev.get(client["dev_type"], [])
    ]

    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(exp, f, sort_keys=False)

    print(f"Generated {filename}")
