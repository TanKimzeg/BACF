import random
import argparse
def get_item_size(data_file):
    lines = open(data_file).readlines()
    item_set = set()
    for line in lines:
        items = line.strip().split()
        items = [int(float(item)) for item in items]
        item_set = item_set | set(items)
    max_item = max(item_set)
    item_size = max_item + 1
    return item_size

def get_user_seqs_and_gene_sample(data_file,item_size):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        items = line.strip().split()
        items = [int(float(item)) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)

    sample_seq = []
    for i in range(len(lines)):
        sample_list = neg_sample(set(user_seq[i]), item_size)
        sample_seq.append(sample_list)

    return sample_seq

def neg_sample(item_set, item_size):  # 前闭后闭
    sample_list = []
    for _ in range(99):
        item = random.random() * (item_size - 1)
        sample_list.append(item)
    return sample_list

def main(args, label:str, dim: tuple[int, int]):
    for node in range(dim[0]):
        for feature in range(dim[1]):
            print(f"Generating neg sample for node {node}, feature {feature} for label {label}")
            try:
                data_file = f"{args.output}/{label}/{node}{feature}.txt"
                item_size = get_item_size(data_file)
                sample_file = f"{args.output}/{label}/{node}{feature}_sample.txt"
                neg_sample = get_user_seqs_and_gene_sample(data_file,item_size)
                output = open(sample_file,'w')
                for i in range(len(neg_sample)):
                    for k in neg_sample[i]:
                        output.write(str(k)+' ')
                    output.write('\n')
                output.close()
            except Exception as e:
                print(f"Error processing node {node}, feature {feature}: {e}")
                continue

if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--data_dir', default='./', type=str)
    # parser.add_argument('--data_name', default='nowplaying', type=str)
    # args = parser.parse_args()
    # args.data_file = args.data_dir + args.data_name + '.txt'
    # args.sample_file = args.data_dir + args.data_name + '_sample.txt'
    # main(args)
