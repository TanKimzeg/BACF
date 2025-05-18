import argparse
from address_check import check_addresses
import graph_layer
import filter_layer
import classifier
import graph_layer.config

def main():
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--addrdir',default=f'F:/json_data/',type=str)
    main_parser.add_argument('--logdir',default=f'F:/log/',type=str)
    main_parser.add_argument('--modelsave',default=f'F:/model_save/',type=str)
    main_parser.add_argument('--output',default=f'F:/output/',type=str)
    main_args:argparse.Namespace = main_parser.parse_args()

    FMLP_parser = argparse.ArgumentParser()
    FMLP_parser.add_argument("--data_dir", default="./data/", type=str)
    FMLP_parser.add_argument("--output_dir", default="F:/output/", type=str)
    FMLP_parser.add_argument("--data_name", default="Beauty", type=str)
    FMLP_parser.add_argument("--do_eval", action="store_true")
    FMLP_parser.add_argument("--load_model", default=None, type=str)

    # model args
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

    # train args
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
    labels = [
        'gambling'
    ]
    for label in labels:
        check_addresses(main_args,label)
        graph_layer.config.args = main_args
        graph_layer.config.label = label
        graph_layer.train2impl(main_args,label)
        filter_layer.generate_sample(main_args,label)
        FMLP_args.data_dir = main_args.output + label + '/'
        FMLP_args.output_dir = main_args.output + label + '/'
        for node in range(8):
            for feature in range(4):
                print(f"Dealing {node}{feature}...")
                FMLP_args.data_name = f"{node}{feature}"
                try: # TODO:值全为0的文件会抛出错误,修改neg_sample
                    filter_layer.FMLP(FMLP_args)
                except Exception as e:
                    print(e)
                    continue
    
    for label in labels:
        # 正类与负类，分别使用LR训练
        classifier.LR(main_args,label,labels)

# if __name__ == '__main__':
main()
    