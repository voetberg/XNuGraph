import click


@click.group("explainer")
@click.pass_context
def explainer(cxt, checkpoint, data_path, device):
    ""


@explainer.command()
def filtered(ctx):
    ""


def edge():
    ""


@explainer.command("shap filtered")
def shap_filtered():
    ""


if __name__ == "__main__":
    explainer()

    explainer.subcommand_metavar()
