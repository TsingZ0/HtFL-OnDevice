import copy
import os
import shutil
import yaml

template_experiment = {
    "name": "benchmark_HtFL_w_same_power_eff",
    "project": "htfl-ondevice",
    "code": {
        "path": "../../../../../", # root folder
        "client": {
            "command": (
                "./config_device_data.sh ${COLEXT_DATASETS}/iWildCam ${COLEXT_CLIENT_ID} && "
                "python3 -m system.clients.client{{ FL_ALGORITHM }} "
                "--server_address=${COLEXT_SERVER_ADDRESS} "
                "--num_classes=158"
            )
        },
        "server": {
            "command": (
                "python3 -m system.servers.server{{ FL_ALGORITHM }} "
                "--min_fit_clients=${COLEXT_N_CLIENTS} "
                "--min_available_clients=${COLEXT_N_CLIENTS} "
                "--num_rounds=10"
            )
        },
    },
    "clients": [
        {"dev_type": "JetsonAGXOrin",    "count": 2, "add_args": "--model=ResNet101"},
        {"dev_type": "JetsonOrinNano",   "count": 2, "add_args": "--model=ResNet152"},
        {"dev_type": "JetsonXavierNX",   "count": 2, "add_args": "--model=ResNet50"},
        {"dev_type": "JetsonNano",       "count": 2, "add_args": "--model=ResNet34"},
        {"dev_type": "OrangePi5B",       "count": 2, "add_args": "--model=ResNet10"},
        # {"dev_type": "LattePandaDelta3", "count": 2, "add_args": "--model=?"}, # Facing power measurement issues
    ],
}

HtFL_algorithms = [
    {"type": "group-heterogeneity", "algorithms": []},
    {"type": "partial-heterogeneity", "algorithms": ["LG", "Gen", "GH"]},
    {"type": "full-heterogeneity", "algorithms": ["FD", "FML", "KD",  "Proto", "TGP"]}
]

# Clean existing output dir before we create new configs
script_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.join(script_dir, "output", "colext_configs")
if os.path.exists(config_dir):
    shutil.rmtree(config_dir)
os.makedirs(config_dir)

# Generate YAML files for each algorithm
config_id = 0
for algorithm_group in HtFL_algorithms:
    algorithm_type = algorithm_group["type"]
    algorithms = algorithm_group["algorithms"]
    for algorithm in algorithms:
        filename = os.path.join(
            config_dir,
            f"{config_id}_{algorithm_type}_{algorithm}.yaml"
        )

        exp = copy.deepcopy(template_experiment)
        for target in ["client", "server"]:
            exp["code"][target]["command"] = (
                exp["code"][target]["command"].replace("{{ FL_ALGORITHM }}", algorithm)
            )

        if algorithm == "Gen":
            exp["code"]["server"]["command"] += " --num_classes=158"

        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(exp, f, sort_keys=False)

        print(f"Generated {filename}")

        config_id += 1
