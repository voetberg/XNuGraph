import click

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


@cli.command("evaluate")
def evaluate():
    ""


if __name__ == "__main__":
    cli()
