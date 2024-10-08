from nugraph.explain_graph import (
    GlobalGNNExplain,
    GNNExplainerDifference,
    FilteredExplainEdges,
    FilteredExplainedHits,
    GNNExplainerHits,
)

from datetime import datetime
import torch
import argparse


explanations = {
    "Edges": GlobalGNNExplain,
    "Hits": GNNExplainerHits,
    "Difference": GNNExplainerDifference,
    "FilterEdge": FilteredExplainEdges,
    "FilterNode": FilteredExplainedHits,
}


def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Trained model checkpoint to test",
        default="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=list(explanations.keys()),
        help="Name of the explaination algorithm",
        default="GNNExplainer",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        type=str,
        help="Full path to output file",
        default="./results/",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5",
        help="Location of input data file",
    )

    parser.add_argument("--test", "-t", action="store_true", help="")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument(
        "--n_batches",
        "-b",
        type=int,
        help="Run only a subset of possible batches",
        default=None,
    )
    parser.add_argument(
        "--message_passing_steps",
        "-s",
        nargs="+",
        type=int,
        help="Different message passing steps to use - runs an explanation for each",
        default=[5],
    )
    return parser.parse_args()


def run_explanation(
    checkpoint, algorithm, outfile, data_path, test, message_passing_steps
):
    file_name = str(datetime.now().timestamp())
    if test:
        file_name = "test"

    for n_steps in message_passing_steps:
        exp_outfile = f"{outfile.rstrip('/')}/{algorithm}_{n_steps}_steps/{file_name}/"

        torch.autograd.set_detect_anomaly(True)
        explain = explanations[algorithm](
            data_path=data_path,
            out_path=exp_outfile,
            message_passing_steps=n_steps,
            checkpoint_path=checkpoint,
            test=test,
        )

        e = explain.explain(explain.data)
        explain.visualize(e, file_name=file_name)
        try:
            explain.save()
        except (NotImplementedError, AttributeError):
            pass


if __name__ == "__main__":
    args = configure()
    run_explanation(
        args.model,
        args.algorithm,
        args.outfile,
        args.data_path,
        args.test,
        args.message_passing_steps,
    )
