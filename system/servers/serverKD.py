import argparse
import os
import flwr as fl
from flwr.common.logger import log
from logging import WARNING, INFO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def weighted_metrics_avg(eval_metrics):
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in eval_metrics)
    metrics_aggregated = {}
    for num_examples, metrics in eval_metrics:
        for key, value in metrics.items():
            if key not in metrics_aggregated:
                metrics_aggregated[key] = num_examples * value
            else:
                metrics_aggregated[key] += num_examples * value
    for key in metrics_aggregated:
        metrics_aggregated[key] /= num_total_evaluation_examples
        log(INFO, f"{key}: {metrics_aggregated[key]}")
    return metrics_aggregated


def recover(compressed_param):
    for k in compressed_param.keys():
        if len(compressed_param[k]) == 3:
            # use np.matmul to support high-dimensional CNN param
            compressed_param[k] = np.matmul(
                compressed_param[k][0] * compressed_param[k][1][..., None, :], 
                    compressed_param[k][2])
    return compressed_param

    
def decomposition(param_iter, energy):
    compressed_param = {}
    for name, param in param_iter:
        try:
            param_cpu = param.detach().cpu().numpy()
        except:
            param_cpu = param
        # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
        if len(param_cpu.shape)>1 and param_cpu.shape[0]>1 and 'embeddings' not in name:
            u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
            # support high-dimensional CNN param
            if len(u.shape)==4:
                u = np.transpose(u, (2, 3, 0, 1))
                sigma = np.transpose(sigma, (2, 0, 1))
                v = np.transpose(v, (2, 3, 0, 1))
            threshold=0
            if np.sum(np.square(sigma))==0:
                compressed_param_cpu=param_cpu
            else:
                for singular_value_num in range(len(sigma)):
                    if np.sum(np.square(sigma[:singular_value_num]))>energy*np.sum(np.square(sigma)):
                        threshold=singular_value_num
                        break
                u=u[:, :threshold]
                sigma=sigma[:threshold]
                v=v[:threshold, :]
                # support high-dimensional CNN param
                if len(u.shape)==4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    sigma = np.transpose(sigma, (1, 2, 0))
                    v = np.transpose(v, (2, 3, 0, 1))
                compressed_param_cpu=[u,sigma,v]
        elif 'embeddings' not in name:
            compressed_param_cpu=param_cpu

        compressed_param[name] = compressed_param_cpu
        
    return compressed_param


# TODO: Implement the server compression logic
# TODO: Implement the energy evolution logic


if __name__ == "__main__":
    # Configration of the server
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--min_fit_clients", type=int, default=2)
    parser.add_argument("--min_available_clients", type=int, default=2)
    args = parser.parse_args()

    # Start server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=FedKD(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            fit_metrics_aggregation_fn=None,
            evaluate_metrics_aggregation_fn=weighted_metrics_avg,
            evaluate_fn=None,
            on_fit_config_fn=None,
            on_evaluate_config_fn=None,
            inplace=True,
            args=args, 
        ),
    )

