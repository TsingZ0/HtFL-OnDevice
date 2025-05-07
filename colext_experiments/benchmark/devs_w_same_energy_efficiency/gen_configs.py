import copy
import os
from pathlib import Path
import shutil
import yaml

template_experiment = {
    "name": "benchmark_HtFL_w_same_power_eff",
    "project": "htfl-ondevice",
    "code": {
        "path": "../../../../../", # root folder
        "client": {
            "command": (
                "./config_device_data.sh ${COLEXT_DATASETS}/iWildCam identity && "
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
    ("Group_Het", []),
    ("Partial_Het", ["LG", "Gen", "GH"]),
    ("Full_Het", ["FD", "FML", "KD",  "Proto", "TGP"])
]

# Clean existing output dir before we create new configs
script_dir = Path(__file__).resolve().parent
config_dir = script_dir / "output" / "colext_configs"
if config_dir.exists():
    shutil.rmtree(config_dir)
config_dir.mkdir()

# Generate YAML files for each algorithm
config_id = 0
for (algorithm_type, algorithms) in HtFL_algorithms:
    for algorithm in algorithms:
        filename = os.path.join(
            config_dir,
            f"{config_id}_{algorithm}-{algorithm_type}.yaml"
        )

        exp = copy.deepcopy(template_experiment)
        for target in ["client", "server"]:
            exp["code"][target]["command"] = (
                exp["code"][target]["command"].replace("{{ FL_ALGORITHM }}", algorithm)
            )

        if algorithm in ["Gen", "GH", "TGP"]:
            exp["code"]["server"]["command"] += " --num_classes=158"

        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(exp, f, sort_keys=False)

        print(f"Generated {filename}")

        config_id += 1
