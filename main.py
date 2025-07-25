import argparse
import os
from address_check import check_addresses
import filter_layer
import classifier
import GFN_layer

def main():
    graph_layer_parser = argparse.ArgumentParser()
    graph_layer_parser.add_argument('--addrdir',default=f'F:/json_data/',type=str)
    graph_layer_parser.add_argument('--logdir',default=f'F:/log/',type=str)
    graph_layer_parser.add_argument('--output',default=f'F:/output/',type=str)
    graph_layer_args:argparse.Namespace = graph_layer_parser.parse_args()

    FMLP_parser = argparse.ArgumentParser()
    FMLP_parser.add_argument("--data_dir", default="./data/", type=str)
    FMLP_parser.add_argument("--output_dir", default="F:/output/", type=str)
    FMLP_parser.add_argument("--data_name", default="Beauty", type=str)
    FMLP_parser.add_argument("--do_eval", action="store_true")
    FMLP_parser.add_argument("--load_model", default=None, type=str)
    FMLP_parser.add_argument("--model_name", default="FMLPRec", type=str)
    FMLP_parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
    FMLP_parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
    FMLP_parser.add_argument("--num_attention_heads", default=2, type=int)
    FMLP_parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    FMLP_parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    FMLP_parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    FMLP_parser.add_argument("--initializer_range", default=0.02, type=float)
    FMLP_parser.add_argument("--max_seq_length", default=50, type=int)
    FMLP_parser.add_argument("--no_filters", action="store_true", help="if no filters, filter layers transform to self-attention")
    FMLP_parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    FMLP_parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    FMLP_parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    FMLP_parser.add_argument("--no_cuda", action="store_true")
    FMLP_parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    FMLP_parser.add_argument("--full_sort", action="store_true")
    FMLP_parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")
    FMLP_parser.add_argument("--seed", default=42, type=int)
    FMLP_parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    FMLP_parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    FMLP_parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    FMLP_parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    FMLP_parser.add_argument("--variance", default=5, type=float)
    FMLP_args = FMLP_parser.parse_args()

    classifier_parser = argparse.ArgumentParser()
    classifier_parser.add_argument('--data_dir', default="F:/output/", type=str)
    classifier_parser.add_argument('--modelsave', default='F:/model_save/', type=str)
    classifier_parser.add_argument('--batch_size', default=32, type=int)
    classifier_parser.add_argument('--lr', default=0.001, type=float)
    classifier_parser.add_argument('--epochs', default=100, type=int)
    classifier_args = classifier_parser.parse_args()

    labels = [
        'Blackmail',
        'Cyber-Security',
        'DarknetMarket',
        'Exchange',
        'P2PFIS',
        'P2PFS',
        'Gambling',    
        'CriminalBlacklist',  
        'MoneyLaundering', 
        'PonziScheme', 
        'MiningPool',  
        'Tumbler',     
        'Individual'   
    ]
    for label in labels:
        check_addresses(graph_layer_args,label)
        GFN_layer.config.args = graph_layer_args
        GFN_layer.config.label = label
        dim_0, dim_1 = GFN_layer.gfn_process(graph_layer_args, label)
        filter_layer.generate_sample(graph_layer_args,label, dim=(dim_0, dim_1))
        FMLP_args.data_dir = os.path.join(graph_layer_args.output, label) + '/'
        FMLP_args.output_dir = os.path.join(graph_layer_args.output, label) + '/'
        for node in range(dim_0):
            for feature in range(dim_1):
                print(f"Dealing {node}{feature}...")
                FMLP_args.data_name = f"{node}{feature}"
                filter_layer.FMLP(FMLP_args)
    
    classifier.LSTM(classifier_args,labels=labels, dim=(dim_0, dim_1))
    classifier.DT(classifier_args,labels=labels, dim=(dim_0, dim_1))
    classifier.KNN(classifier_args,labels=labels, dim=(dim_0, dim_1))
    classifier.LightGBM(classifier_args,labels=labels, dim=(dim_0, dim_1))
    classifier.LR(classifier_args,labels=labels, dim=(dim_0, dim_1))
    classifier.RF(classifier_args,labels=labels, dim=(dim_0, dim_1))
    classifier.SVM(classifier_args,labels=labels, dim=(dim_0, dim_1))
    classifier.XGBoost(classifier_args,labels=labels, dim=(dim_0, dim_1))

if __name__ == '__main__':
    main()
    