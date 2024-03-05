import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from nugraph.explain_graph.explainer_probes import ExplainNetwork, DynamicExplainNetwork
from nugraph.explain_graph.gnn_explain import GlobalGNNExplain, ClasswiseGNNExplain, GNNExplainFeatures,  GNNExplainerPrune
import argparse
from datetime import datetime

explainations = {
    "GNNExplainer":GlobalGNNExplain, 
    "ProbeExplainer": ExplainNetwork,
    "GNNFeatures": GNNExplainFeatures,
    "GNNExplainClass": ClasswiseGNNExplain, 
    "DynamicProbeExplainer": DynamicExplainNetwork,
    "GNNExplainerPrune": GNNExplainerPrune
}

def configure(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", '-c', type=str, help='Trained model checkpoint to test', default="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt")
    parser.add_argument("--algorithm", '-a', type=str, 
                        choices=list(explainations.keys()), 
                        help='Name of the explaination algorithm', default="GNNExplainer")
    parser.add_argument("--outfile", '-o', type=str, help='Full path to output file', default="./results/")
    parser.add_argument('--data_path', '-d', type=str,
                          default="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5",
                          help='Location of input data file')

    parser.add_argument("--test", '-t', action="store_true", help='')
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--n_batches", "-b", type=int, help='Run only a subset of possible batches', default=None)
    parser.add_argument("--message_passing_steps", "-m",
                         nargs='+', 
                         type=int, 
                         help="Different message passing steps to use - runs an explaination for each", 
                         default=[5])
    return parser.parse_args()

def run_explaination(checkpoint, algorithm, outfile, data_path, test, message_passing_steps): 
    file_name = str(datetime.now().timestamp())
    if test: 
        file_name = "test"
        
    for n_steps in message_passing_steps: 
        exp_outfile = f"{outfile.rstrip('/')}/{algorithm}_{n_steps}_steps/"

        explain = explainations[algorithm](
            data_path=data_path, 
            out_path=exp_outfile, 
            message_passing_steps=n_steps, 
            checkpoint_path=checkpoint, 
            test=test)

        e = explain.explain(explain.data)
        explain.visualize(e, file_name=file_name)
        try: 
            explain.save()
        except: 
            pass 

if __name__=='__main__': 
    args = configure()
    run_explaination(args.checkpoint, args.algorithm, args.outfile, args.data_path, args.test, args.message_passing_steps) 