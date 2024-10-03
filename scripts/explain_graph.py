import click
import os
from datetime import datetime

import torch

from nugraph.explain_graph import GNNExplainerHits, FilteredExplainEdges


@click.group("explainer")
@click.pass_context
@click.option("-c", "--checkpoint", default="./paper.ckpt")
@click.option("-d", "--data-path", default="./test_data/test_data.h5")
@click.option("-o", "--out-path", default="./test/")
@click.option("-t", "--plot-title", default="")
@click.option("--gpu/--no-gpu")
def explainer(ctx, checkpoint, data_path, out_path, plot_title, gpu):
    class config(object):
        def __init__(self, checkpoint, data_path, out_path, plot_title, gpu) -> None:
            self.out_path = out_path
            self.checkpoint = checkpoint
            self.data_path = data_path
            self.plot_title = plot_title
            self.device = "cuda" if gpu else "cpu"

    ctx.obj = config(checkpoint, data_path, out_path, plot_title, gpu)
    if not os.path.exists(out_path):
        os.makedirs(out_path)


@explainer.command()
@click.pass_context
@click.option("-m", "--message-passing-steps", default=5)
@click.option("-e", "--epochs", default=50)
def hits(ctx, message_passing_steps, epochs):
    file_name = f"hit_explanation_{str(datetime.now().timestamp())}"
    if ctx.obj.plot_title != "":
        file_name = ctx.obj.plot_title.lower().replace(" ", "_") + file_name

    exp_outfile = f"{ctx.obj.out_path.rstrip('/')}/{file_name}/"

    torch.autograd.set_detect_anomaly(True)
    explain = GNNExplainerHits(
        data_path=ctx.obj.data_path,
        out_path=exp_outfile,
        message_passing_steps=message_passing_steps,
        checkpoint_path=ctx.obj.checkpoint,
        n_epochs=epochs,
        node_attribution=False,
    )

    e = explain.explain(explain.data)
    explain.visualize(e, file_name=file_name)
    try:
        explain.save()
    except (NotImplementedError, AttributeError):
        pass


@explainer.command()
@click.pass_context
@click.option("-m", "--message-passing-steps", default=5)
@click.option("-e", "--epochs", default=50)
def features(ctx, message_passing_steps, epochs):
    file_name = f"feature_explanation_{str(datetime.now().timestamp())}"
    if ctx.obj.plot_title != "":
        file_name = ctx.obj.plot_title.lower().replace(" ", "_") + file_name

    exp_outfile = f"{ctx.obj.out_path.rstrip('/')}/{file_name}/"

    torch.autograd.set_detect_anomaly(True)
    explain = GNNExplainerHits(
        data_path=ctx.obj.data_path,
        out_path=exp_outfile,
        message_passing_steps=message_passing_steps,
        checkpoint_path=ctx.obj.checkpoint,
        n_epochs=epochs,
        node_attribution=True,
    )

    e = explain.explain(explain.data)
    explain.visualize(e, file_name=file_name)
    try:
        explain.save()
    except (NotImplementedError, AttributeError):
        pass


@explainer.command()
@click.option("-m", "--message-passing-steps", default=5)
@click.option("-e", "--epochs", default=50)
@click.pass_context
def edge(ctx, message_passing_steps, epochs):
    file_name = f"edge_explanation_{str(datetime.now().timestamp())}"
    if ctx.obj.plot_title != "":
        file_name = ctx.obj.plot_title.lower().replace(" ", "_") + file_name

    exp_outfile = f"{ctx.obj.out_path.rstrip('/')}/{file_name}/"

    torch.autograd.set_detect_anomaly(True)
    explain = FilteredExplainEdges(
        data_path=ctx.obj.data_path,
        out_path=exp_outfile,
        message_passing_steps=message_passing_steps,
        checkpoint_path=ctx.obj.checkpoint,
        n_epochs=epochs,
    )

    e = explain.explain(explain.data)
    explain.visualize(e)
    try:
        explain.save()
    except (NotImplementedError, AttributeError):
        pass


if __name__ == "__main__":
    explainer()
