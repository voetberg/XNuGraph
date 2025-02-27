import click
import os

from torch_geometric.loader import DataLoader

from nugraph.explain_graph.algorithms.latent_representation import (
    LatentRepresentation,
)
from nugraph.explain_graph.utils import Load, network_forward


def embed(network_step, message_steps, model):
    planes = {"u", "v", "y"}
    return {
        "encoder": lambda x,: network_forward.encoder_forward(model, planes, x),
        "message": lambda x: network_forward.message_forward(
            model, planes, x, message_steps
        ),
    }[network_step]


@click.group()
@click.option("-c", "--checkpoint", default="./paper.ckpt")
@click.option("-d", "--data-path", default="./test_data/test_data.h5")
@click.option("-o", "--out-path", default="./test/")
@click.option("-t", "--plot-title", default="")
@click.option("--test/--no-test", default=False)
@click.option("--gpu/--no-gpu")
@click.pass_context
def cli(ctx, checkpoint, data_path, out_path, plot_title, test, gpu):
    class config(object):
        def __init__(
            self, checkpoint, data_path, out_path, plot_title, test, gpu
        ) -> None:
            self.out_path = out_path
            self.load = Load(
                checkpoint_path=checkpoint,
                data_path=data_path,
                load_data=True,
                load_model=True,
                test=test,
                auto_load=True,
            )
            self.plot_title = plot_title
            self.device = "cuda" if gpu else "cpu"

    ctx.obj = config(checkpoint, data_path, out_path, plot_title, test, gpu)
    if not os.path.exists(out_path):
        os.makedirs(out_path)


@cli.command("decompose")
@click.option(
    "-s", "--network-step", type=click.Choice({"message", "encoder", "decoder"})
)
@click.option("-m", "--message-passing-steps", default=0)
@click.option("-t", "--data-threshold", default=None, type=float)
@click.pass_context
def decompose(ctx, network_step, message_passing_steps, data_threshold):
    # Do the latent decomposition and do not cluster the output
    name = f"{network_step}_{message_passing_steps}_clustering"
    data = DataLoader(ctx.obj.load.data, shuffle=True)
    rep = LatentRepresentation(
        embedding_function=embed(
            network_step=network_step,
            message_steps=message_passing_steps,
            model=ctx.obj.load.model,
        ),
        data_loader=data,
        out_path=ctx.obj.out_path,
        name=name,
        title=ctx.obj.plot_title,
        batched_fit=True,
        n_threshold=data_threshold,
    )
    rep.decompose()
    rep.visualize_label_silhouette()
    rep.save_results()


if __name__ == "__main__":
    cli()
