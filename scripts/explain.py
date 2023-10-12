from nugraph.explain_graph.gnn_explain import GNNExplain
import argparse

explainations = {
    "GNNExplainer":GNNExplain
}

def configure(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", '-c', type=str, help='Trained model checkpoint to test')
    parser.add_argument("--algorithm", '-a', type=str, 
                        choices=list(explainations.keys()), 
                        help='Name of the explaination algorithm')
    parser.add_argument("--outfile", '-o', type=str, help='Full path to output file')
    parser.add_argument('--data_path', type=str,
                          default='/raid/uboone/CHEP2023/CHEP2023.gnn.h5',
                          help='Location of input data file')
    parser.add_argument('--batch_size', type=int, default=64,
                          help='Size of each batch of graphs')
    parser.add_argument("--test", '-t', type=bool, default=False, help='')
    return parser.parse_args()

def run_explaination(checkpoint, algorithm, outfile, data_path, batch_size, test): 
    if test: 
        outfile = f"{outfile.rstrip('/')}/test/"

    explain = explainations[algorithm](
        data_path=data_path, 
        out_path=outfile, 
        checkpoint_path=checkpoint, 
        batch_size=batch_size)

    if test: 
        data = explain.data.val_dataloader.get(0)
        e = explain.explain(data, raw=True, node_index=[1])
        explain.visualize(e, file_name="test.png")
        
    else: 
        #TODO Nodes to explain for each graph - or a different data framework entirely
        explain.visualize()
        explain.save()

if __name__=='__main__': 
    args = configure()
    run_explaination(**args)