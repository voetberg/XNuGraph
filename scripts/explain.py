from nugraph.explain_graph.gnn_explain import GlobalGNNExplain
from nugraph.explain_graph.gnn_explain_features import GNNExplainFeatures
import argparse

explainations = {
    "GNNExplainer":GlobalGNNExplain, 
    "GNNFeatures": GNNExplainFeatures
}

def configure(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", '-c', type=str, help='Trained model checkpoint to test', default="./paper.ckpt")
    parser.add_argument("--algorithm", '-a', type=str, 
                        choices=list(explainations.keys()), 
                        help='Name of the explaination algorithm', default="GNNExplainer")
    parser.add_argument("--outfile", '-o', type=str, help='Full path to output file', default="./")
    parser.add_argument('--data_path', type=str,
                          default='/raid/uboone/CHEP2023/CHEP2023.gnn.h5',
                          help='Location of input data file')
    parser.add_argument('--batch_size', type=int, default=2,
                          help='Size of each batch of graphs')
    parser.add_argument("--test", '-t', action="store_true", help='')
    parser.add_argument("--interactive", "-i", action="store_true")
    return parser.parse_args()

def run_explaination(checkpoint, algorithm, outfile, data_path, batch_size, test, interactive): 
    file_name = None
    if test: 
        outfile = f"{outfile.rstrip('/')}/explainer_test/"
        file_name = "test"

    explain = explainations[algorithm](
        data_path=data_path, 
        out_path=outfile, 
        checkpoint_path=checkpoint, 
        batch_size=batch_size, 
        test=test)
    
    e = explain.explain(explain.data)
    explain.visualize(e, file_name=file_name, interactive=interactive)
    explain.save(file_name=file_name)

if __name__=='__main__': 
    args = configure()
    run_explaination(args.checkpoint, args.algorithm, args.outfile, args.data_path, args.batch_size, args.test, args.interactive) 