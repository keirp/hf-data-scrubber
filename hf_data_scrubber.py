import argparse
from datasets import load_dataset
from nltk.util import ngrams
import jsonlines
from tqdm import tqdm
import re
import os

def split_on_contamination(data_batch, test_ngrams, args):
    # Data batch comes as a dict with list values. Let's reorganize it as a list of dicts
    # data_batch = [{k: v[i] for k, v in data_batch.items()} for i in range(len(data_batch[args.column]))]
    # Compute 13-grams for the batch
    ngrams_batch = []
    for txt in data_batch[args.column]:
        obj_ngrams = ngrams(txt.split(), args.n)
        obj_ngrams = [' '.join(ngram) for ngram in obj_ngrams]
        ngrams_batch.append(set(obj_ngrams))
    
    decontaminated_batch = {k: [] for k in data_batch.keys()}

    # Compare the ngrams of the batch with the ngrams of the test set
    for i in range(len(data_batch)):
        if ngrams_batch[i].intersection(test_ngrams):
            # print('Found contamination in text: ', data_batch[args.column][i])
            print('Contaminating ngrams: ', ngrams_batch[i].intersection(test_ngrams))
            # Split the text on the ngrams in the intersection
            # First, create a regex to split on the ngrams
            regex = '|'.join(re.escape(ngram) for ngram in ngrams_batch[i].intersection(test_ngrams))
            # Split the text
            split_text = re.split(regex, data_batch[args.column][i])
            # print('Split text: ', split_text)
            # Now for each of these splits, add a new item to the batch
            for txt in split_text:
                decontaminated_batch[args.column].append(txt)
                for k in data_batch.keys():
                    if k != args.column:
                        decontaminated_batch[k].append(data_batch[k][i])
        else:
            for k in data_batch.keys():
                decontaminated_batch[k].append(data_batch[k][i])

    return decontaminated_batch  

def main(args):
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)
    # Load test_set
    with jsonlines.open(args.test_set) as reader:
        test_set = [obj for obj in reader]

    # Compute 13-grams for the test set
    print('Computing 13-grams for the test set')
    test_ngrams = []
    for obj in tqdm(test_set):
        obj_ngrams = ngrams(obj.split(), args.n)
        obj_ngrams = [' '.join(ngram) for ngram in obj_ngrams]
        test_ngrams.extend(list(obj_ngrams))
    test_ngrams = set(test_ngrams)

    print('Decontaminating dataset')
    decontaminated_dataset = ds.map(lambda x: split_on_contamination(x, test_ngrams, args), batched=True, batch_size=1000, num_proc=args.n_processes)

    print(decontaminated_dataset)

    if args.output_dataset:
        decontaminated_dataset.save_to_disk(os.path.join(args.cache_dir, args.output_dataset))
        decontaminated_dataset.push_to_hub(args.output_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='keirp/open-web-math-hq-dev')
    parser.add_argument('--column', type=str, default='text')
    parser.add_argument('--test_set', type=str, default='data/test.jsonl')
    parser.add_argument('--n', type=int, default=13)
    parser.add_argument('--n_processes', type=int, default=8)
    parser.add_argument('--output_dataset', type=str, default=None)
    parser.add_argument('--cache_dir', type=str)

    args = parser.parse_args()
    main(args)