from nugraph.explain_graph.gnn_explain import GlobalGNNExplain
from nugraph.explain_graph.gnn_explain_features import GNNExplainFeatures
import argparse

explainations = {
    "GNNExplainer":GlobalGNNExplain, 
    "GNNFeatures": GNNExplainFeatures
}

def configure(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", '-c', type=str, help='Trained model checkpoint to test')
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
    return parser.parse_args()

def run_explaination(checkpoint, algorithm, outfile, data_path, batch_size, test): 
    if test: 
        outfile = f"{outfile.rstrip('/')}/explaination_test/"

    explain = explainations[algorithm](
        data_path=data_path, 
        out_path=outfile, 
        checkpoint_path=checkpoint, 
        batch_size=batch_size, 
        test=test)

    if test: 
        data = explain.data
        e = explain.explain(data, raw=True)
        explain.visualize(e, file_name="test")
        
    else: 
        #TODO Nodes to explain for each graph - or a different data framework entirely
        explain.visualize()
        explain.save()

if __name__=='__main__': 
    args = configure()
    run_explaination(args.checkpoint, args.algorithm, args.outfile, args.data_path, args.batch_size, args.test) 