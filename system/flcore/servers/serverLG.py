import argparse
import flwr as fl


if __name__ == "__main__":
    # Configration of the server
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=3)
    args = parser.parse_args()

    # Start server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )

