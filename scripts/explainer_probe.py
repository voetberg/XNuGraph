from collections import defaultdict
import json
import os
import click
import numpy as np
import matplotlib.pyplot as plt

from nugraph.explain_graph.algorithms.linear_probes.concept_activation_vectors import (
    ConceptActivateVectors,
)
from nugraph.explain_graph.algorithms.linear_probes.feature_loss import FeatureLoss
from nugraph.explain_graph.utils.load import Load


@click.group()
def cli():
    pass


def init_concept_probe(
    checkpoint,
    data_path,
    loss_function,
    message_passing_steps,
    network_step,
    gpu,
    name,
    out_path,
):
    if gpu:
        device = "cuda"
        total_devices = 1
    else:
        device = "cpu"
        total_devices = 0

    if name is None:
        name = f"{loss_function}_{network_step}{message_passing_steps}"
    loader = Load(checkpoint_path=checkpoint, data_path=data_path, load_data=True)
    probe_engine = ConceptActivateVectors(
        model=loader.model,
        data=loader.data,
        rank=device,
        probe_name=name,
        out_path=f"{out_path.rstrip('/')}/{name}",
        total_devices=total_devices,
        multicore=False,
        loss_function=loss_function,
    )
    return probe_engine


## Train a probe


@cli.command("concept-probe")
@click.option("-c", "--checkpoint")
@click.option("-d", "--data-path")
@click.option("-e", "--epochs", type=int, default=25)
@click.option(
    "-l",
    "--loss-function",
    type=click.Choice(
        {"michel", "hipmip", "tracks", "michel_energy"}, case_sensitive=False
    ),
)
@click.option("-s", "--network-step", type=click.Choice({"encoder", "message"}))
@click.option("-m", "--message-passing-steps", type=int, default=1)
@click.option("--gpu/--no-gpu", default=False)
@click.option("-n", "--name", default=None)
@click.option("-o", "--out-path", default="./results/concept_probe/")
def train_probe(
    checkpoint,
    data_path,
    epochs,
    loss_function,
    message_passing_steps,
    network_step,
    gpu,
    name,
    out_path,
):
    probe_engine = init_concept_probe(
        checkpoint,
        data_path,
        loss_function,
        message_passing_steps,
        network_step,
        gpu,
        name,
        out_path,
    )
    if network_step == "encoder":
        probe_engine.train_encoder(epochs=epochs, test=False, overwrite=True)
    else:
        probe_engine.train_message(
            epochs=epochs,
            message_step=message_passing_steps,
            test=False,
            overwrite=True,
        )


@cli.command("kfold-probe")
@click.option("-k", "--k-folds", type=int)
@click.option("-c", "--checkpoint")
@click.option("-d", "--data-path")
@click.option("-e", "--epochs", type=int, default=25)
@click.option(
    "-l",
    "--loss-function",
    type=click.Choice(
        {"michel", "hipmip", "tracks", "michel_energy"}, case_sensitive=False
    ),
)
@click.option("-s", "--network-step", type=click.Choice({"encoder", "message"}))
@click.option("-m", "--message-passing-steps", type=int, default=1)
@click.option("--gpu/--no-gpu", is_flag=True)
@click.option("-n", "--name", default=None)
@click.option("-o", "--out-path", default="./results/concept_probe/")
def kfold(
    checkpoint,
    data_path,
    epochs,
    loss_function,
    message_passing_steps,
    network_step,
    gpu,
    name,
    out_path,
    k_folds,
):
    probe_engine = init_concept_probe(
        checkpoint,
        data_path,
        loss_function,
        message_passing_steps,
        network_step,
        gpu,
        name,
        out_path,
    )
    if network_step == "encoder":
        probe_engine.kfold_train(
            embedding_function=probe_engine.encoder_in_func,
            folds=k_folds,
            epochs=epochs,
        )
    else:

        def embedding(x):
            return probe_engine.message_in_function(x, message_passing_steps)

        probe_engine.kfold_train(
            embedding_function=embedding, folds=k_folds, epochs=epochs
        )


## Evaluate


@cli.command("visualize")
@click.option("-p", "--path")
@click.option("-t", "--title")
@click.option("--baseline/--no-baseline", default=False)
@click.option("-c", "--checkpoint")
@click.option("-d", "--data-path")
@click.option(
    "-l",
    "--loss-function",
    type=click.Choice(
        {"michel", "hipmip", "tracks", "michel_energy"}, case_sensitive=False
    ),
)
@click.option("--gpu/--no-gpu", default=False)
@click.option("--baseline-cutoff", default=25)
def visualize(
    path,
    title,
    make_baseline,
    checkpoint,
    data_path,
    loss_function,
    gpu,
    baseline_cutoff,
):
    if make_baseline:
        load = Load(checkpoint_path=checkpoint, data_path=data_path, batch_size=1200)
        subset = load.data
        device = "cuda" if gpu else "cpu"
        loss = FeatureLoss(feature=loss_function, device=device)

        running_loss = []
        for data in subset:
            load.model.step(data)
            y_hat = data.collect("x_semantic")
            y = data.collect("y_semantic")
            loss_step = loss.loss(y_hat, y).item()
            running_loss.append(loss_step)
            if len(running_loss) >= baseline_cutoff:
                continue
        baseline = sum(running_loss) / len(running_loss)
    else:
        baseline = None

    ConceptActivateVectors.visualize(
        show=False, base_path=path, title=title, baseline=baseline
    )


def load_kfold_history(
    results_loc="results/concept_probe/michel", history_name="_probe_history.json"
):
    folders = os.listdir(results_loc)
    histories = defaultdict(list)
    for folder in folders:
        if os.path.isdir(folder):
            continue

        index = [i for i in folder if i.isdigit()]
        try:
            index = int(index[0])
        except (ValueError, IndexError):
            index = 0

        fold_histories = [
            file
            for file in os.listdir(f"{results_loc}/{folder}")
            if history_name in file
        ]
        for file in fold_histories:
            json_history = json.load(open(f"{results_loc}/{folder}/{file}"))
            histories[index].append(json_history)

        histories[index] = np.array(histories[index])
    return histories


def load_history(
    results_loc="results/concept_probe/michel",
    history_name="_probe_history.json",
    kfold=False,
):
    if kfold:
        return load_kfold_history(results_loc, history_name)
    files = [file for file in os.listdir(results_loc) if history_name in file]
    loss_histories = {}
    for file in files:
        f = json.load(open(f"{results_loc}/{file}"))
        index = file.rstrip(history_name).lstrip("message")
        index = [i for i in index if i.isdigit()]
        try:
            index = int(index[0])
        except (ValueError, IndexError):
            index = 0
        loss_histories[index] = f

    return loss_histories


@cli.command("evaluate")
@click.argument("--results_loc")
@click.argument("--change_cutoff")
@click.argument("--baseline")
@click.argument("--history_name")
@click.argument("--baseline_name")
@click.argument("--reference_name")
@click.argument("--title")
@click.argument("--kfold/--no-kfold")
def evaluate(
    results_loc,
    change_cutoff=10 ** (-4),
    baseline=1.597,
    kfold=False,
    history_name="",
    baseline_name="Against Network Decoder",
    reference_name="Against Final Probe",
    title="Michel Concept Probe",
):
    loss_histories = load_history(results_loc, history_name, kfold)
    if not kfold:
        reference_history = loss_histories[max(loss_histories.keys())]
        if baseline is None:
            baseline = min(reference_history)

        history_diff = np.abs(np.diff(np.array(reference_history)))
        try:
            reference_epoch = np.where(history_diff <= change_cutoff)[0][0]
        except IndexError:
            raise ValueError(
                "Change cutoff is too small, increase it to see more reasonable results"
            )

        reference_loss = reference_history[reference_epoch]

        _, vis = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)

        table = defaultdict(dict)
        for index, history in loss_histories.items():
            ref = history[reference_epoch]
            table[baseline_name][index] = ref - baseline
            table[reference_name][index] = ref - reference_loss

        vis[0].set_title(baseline_name)
        vis[0].scatter(table[baseline_name].keys(), table[baseline_name].values())
        vis[0].set_xticks(list(table[baseline_name].keys()))

        vis[1].set_title(reference_name)
        vis[1].scatter(table[reference_name].keys(), table[reference_name].values())
        vis[1].set_xticks(list(table[reference_name].keys()))

        vis[0].hlines(y=0.0, xmin=0, xmax=max(loss_histories.keys()), color="grey")
        vis[1].hlines(y=0.0, xmin=0, xmax=max(loss_histories.keys()), color="grey")

        plt.suptitle(title)
        plt.savefig(f"{results_loc.rstrip('/')}/{title.lower().replace(" ", "_")}.png")
        print(table)

    else:
        evaluate_kfold(loss_histories)


def evaluate_kfold(
    histories,
    results_loc,
    change_cutoff=10 ** (-4),
    baseline=1.597,
    baseline_name="Against Network Decoder",
    reference_name="Against Final Probe",
    title="Michel Concept Probe",
):
    reference_history = np.mean(histories[max(histories.keys())])
    history_diff = np.abs(np.diff(np.array(reference_history)))
    try:
        reference_epoch = np.where(history_diff <= change_cutoff)[0][0]
    except IndexError:
        raise ValueError(
            "Change cutoff is too small, increase it to see more reasonable results"
        )

    if baseline is None:
        baseline = min(reference_history)

    reference_loss = reference_history[reference_epoch]
    table = defaultdict(dict)
    violin = defaultdict(list)
    histories = {k: v for k, v in sorted(histories.items(), key=lambda item: item[0])}
    for index, history in histories.items():
        try:
            kfold = history[:, reference_epoch]

            violin[baseline_name].append(kfold - baseline)
            violin[reference_name].append(kfold - reference_loss)

            table[f"{baseline_name} Mean"][index] = np.mean(kfold - baseline)
            table[f"{baseline_name} Sigma"][index] = np.std(kfold - baseline)

            table[f"{reference_name} Mean"][index] = np.mean(kfold - reference_loss)
            table[f"{reference_name} Sigma"][index] = np.std(kfold - reference_loss)

        except IndexError:
            pass

    _, vis = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    vis[0][0].set_title(baseline_name)
    vis[0][0].errorbar(
        table[f"{baseline_name} Mean"].keys(),
        table[f"{baseline_name} Mean"].values(),
        yerr=list(table[f"{baseline_name} Sigma"].values()),
        fmt="o",
    )
    vis[0][0].set_xticks(list(table[f"{baseline_name} Mean"].keys()))

    vis[0][1].set_title(reference_name)
    vis[0][1].errorbar(
        table[f"{reference_name} Mean"].keys(),
        table[f"{reference_name} Mean"].values(),
        yerr=list(table[f"{reference_name} Sigma"].values()),
        fmt="o",
    )
    vis[0][1].set_xticks(list(table[f"{baseline_name} Mean"].keys()))
    vis[0][0].hlines(
        y=0.0, xmin=0, xmax=max(table[f"{baseline_name} Mean"].keys()), color="grey"
    )
    vis[0][1].hlines(
        y=0.0, xmin=0, xmax=max(table[f"{baseline_name} Mean"].keys()), color="grey"
    )

    for index, row in enumerate(violin[baseline_name]):
        vis[1][0].violinplot(row, positions=[index])
    for index, row in enumerate(violin[reference_name]):
        vis[1][1].violinplot(row, positions=[index])

    vis[1][0].set_xticks(list(table[f"{baseline_name} Mean"].keys()))
    vis[1][1].set_xticks(list(table[f"{baseline_name} Mean"].keys()))

    plt.suptitle(title)
    plt.savefig(
        f"{results_loc.rstrip('/')}/{title.lower().replace(" ", "_")}_kfold.png"
    )

    print(table)


if __name__ == "__main__":
    cli()


# Example  -
# kfold-probe -k 5 -c ../nugraph_resources/paper.ckpt -d /nugraph/NG2-paper.gnn.h5 -o ../results/kfold_probe/michel -n message1 -s message -m 1 -l michel
