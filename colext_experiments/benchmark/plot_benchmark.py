import os
import argparse
import subprocess
from matplotlib import pyplot as plt, rcParams
import seaborn as sns
import pandas as pd
import numpy as np

rcParams["savefig.format"] = 'png'
OUTPUT_FORMAT_CONFIG = {"bbox_inches": 'tight', "dpi": 300}

def make_plots(job_ids_file, benchmark_output, args):
    plots_dir = os.path.join(benchmark_output, "plots")
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    job_summaries_df, round_metrics_df = prepare_data(job_ids_file, benchmark_output, args.force_collect)

    plot_totals(round_metrics_df, job_summaries_df, plots_dir)

    dev_order = ["JetsonAGXOrin", "JetsonOrinNano", "JetsonXavierNX", "JetsonNano", "JetsonXavierNX", "OrangePi5B"]
    header_cols = ["Training time ps (ms)", "Energy ps (mJ)"]
    cat_plot(job_summaries_df, plots_dir, "per_dev_ps", header_cols, x_col="dev_type", hue_col="job_id",
             extra_plot_args={
                 "order": dev_order
                 }
            )

    header_cols = ["Training time ps (ms)", "Energy ps (mJ)"]
    cat_plot(job_summaries_df, plots_dir, "per_dev_ps_zoom_jon", header_cols, x_col="dev_type", hue_col="job_id",
                y_limit_dev="JetsonOrinNano",
                extra_plot_args = {
                    "order": dev_order,
                    "errorbar": None,
                    "estimator": np.mean
                    }
                )

    header_cols = ["Round time (s)", "Training time (s)", "Energy in round (J)"]
    cat_plot(job_summaries_df, plots_dir, "per_job_round", header_cols, x_col="job_id", hue_col="dev_type",
                # extra_plot_args = {
                #     "errorbar": None,
                #     "estimator": np.max
                #     }
                 )

def prepare_data(job_ids_file, benchmark_output, force_collect=False):
    print(f"Reading job ids from '{job_ids_file}'")
    with open(job_ids_file, "r", encoding="utf-8") as f:
        job_id_map = dict(line.strip().split('=', 1) for line in f.readlines())
    job_ids = job_id_map.keys()

    collect_job_metrics(job_ids, benchmark_output, force_collect)
    job_summaries_df = read_colext_metric_file_as_df("client_rounds_summary.csv", job_ids, benchmark_output)
    round_metrics_df = read_colext_metric_file_as_df("round_metrics.csv", job_ids, benchmark_output)

    # Map job id to experiment config file name. Ex: 1209=ResNet18
    job_summaries_df["job_id"] = job_summaries_df["job_id"].map(job_id_map)
    round_metrics_df["job_id"] = round_metrics_df["job_id"].map(job_id_map)

    return job_summaries_df, round_metrics_df

def collect_job_metrics(job_ids, output_parent_dir, force_collect=False):
    for job_id in job_ids:
        job_metrics_dir = os.path.join(output_parent_dir, "colext_metrics", job_id)

        if os.path.isdir(job_metrics_dir) and not force_collect:
            print(f"Skipping job_id = {job_id} as metrics already collected")
            continue

        command = ["colext_get_metrics", "-j", str(job_id)]
        if force_collect:
            command += ["-f"]

        result = subprocess.run(command, cwd=output_parent_dir, check=True)
        if result.returncode != 0:
            print(f"ERROR: Could not collect job metrics for job_id = {job_id}")

def read_colext_metric_file_as_df(metric_file, job_ids, job_metrics_parent_dir):
    print(f"Reading metrics from {metric_file} and merging as df")
    result_df = []
    for job_id in job_ids:
        file_path = os.path.join(
            job_metrics_parent_dir,
            "colext_metrics",
            job_id,
            "raw",
            metric_file
        )
        df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])
        df = df.assign(job_id=job_id)
        result_df.append(df)

    return pd.concat(result_df, ignore_index=True)

def cat_plot(job_summaries_df, plots_dir, name_suffix, header_cols,
                 x_col="dev_type", hue_col="job_id", y_limit_dev=None, extra_plot_args={}):
    # Prepare dataset for plotting
    id_vars=["dev_type", "job_id", "stage"]
    cols = header_cols + id_vars
    job_summaries_df = job_summaries_df[cols]
    df_long = pd.melt(job_summaries_df, id_vars=id_vars, var_name='metric')

    g = sns.catplot(x=x_col, y="value", hue=hue_col, data=df_long,
                    col="metric", row="stage",
                    row_order=["FIT", "EVAL"],
                    kind="bar", sharey=False, height=3,
                    **extra_plot_args)
                    # hue_order=sorted(df_long["job_id"].unique())

    g.set_axis_labels("", "")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.figure.autofmt_xdate(rotation=70)
    # for ax in g.axes.flat:
    #     ax.tick_params(axis='x', labelrotation=70)

    # Set y axis limit as the max median value bench device * max_increase
    if y_limit_dev:
        bench_dev_df = job_summaries_df[job_summaries_df["dev_type"] == y_limit_dev]
        max_increase = 1.7
        for ax in g.axes.flat:
            (stage, col) = ax.get_title().split(' | ', 1)
            max_median_value = bench_dev_df[bench_dev_df["stage"] == stage].groupby(hue_col)[col].mean().max()
            ax.set_ylim(0, max_median_value * max_increase)
            ax.axhline(y=max_median_value, color='k', linestyle='--', alpha=0.7)

    output_file = os.path.join(plots_dir, f"cat_plot_{name_suffix}")
    g.figure.savefig(f"{output_file}", **OUTPUT_FORMAT_CONFIG)

def plot_totals(round_metrics_df, job_summaries_df, plots_dir):
    orig_order = round_metrics_df["job_id"].unique()

    total_exec_time_df = round_metrics_df.groupby("job_id").apply(
        lambda x: pd.Series({"Execution Time (s)": (x.iloc[-1]["end_time"] - x.iloc[0]["start_time"]).total_seconds()})
    )

    total_energy_df = job_summaries_df.groupby("job_id").apply(
        lambda x: pd.Series({"Total Energy (kJ)": x["Energy in round (J)"].sum() / 1000})
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})
    sns.barplot(x="job_id", y="Execution Time (s)", data=total_exec_time_df, ax=ax1, order=orig_order)
    sns.barplot(x="job_id", y="Total Energy (kJ)", data=total_energy_df, ax=ax2, order=orig_order)
    fig.autofmt_xdate(rotation=70)

    fig.savefig(os.path.join(plots_dir, "per_job"), **OUTPUT_FORMAT_CONFIG)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_dir", type=str, help="Directory with benchmark to plot")
    parser.add_argument("-p", "--plots_dir", type=str, default="plots", help="Output dir for plots")
    parser.add_argument("-f", "--force_collect", action="store_true", default=False, help="Force collection of metrics even if dir exists")
    args = parser.parse_args()

    if not os.path.isdir(args.bench_dir):
        print(f"ERROR: Benchmark dir '{args.bench_dir}' does not exist")
        exit(1)

    benchmark_output = os.path.join(args.bench_dir, "output")
    if not os.path.isdir(benchmark_output):
        print(f"ERROR: Benchmark output dir '{benchmark_output}' not found.")
        print("Have you ran the benchmark?")
        exit(1)

    output_job_ids_files = [f for f in os.listdir(benchmark_output) if f.startswith("output_job_id_maps")]
    if len(output_job_ids_files) < 1:
        print(f"ERROR: Could not find file 'output_job_id_maps' in {benchmark_output}")
        print("Have you ran the benchmark?")
        exit(1)

    if len(output_job_ids_files) > 1:
        output_job_ids_files.sort(reverse=True)
        print(f"WARNING: Found more than one file 'output_job_id_maps' in {benchmark_output}. Picking the latest one.")

    output_job_ids_file = os.path.join(benchmark_output, output_job_ids_files[0])
    make_plots(output_job_ids_file, benchmark_output, args)

if __name__ == "__main__":
    main()
