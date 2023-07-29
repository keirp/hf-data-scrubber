import argparse
from datasets import load_dataset
from nltk.util import ngrams
import jsonlines
from tqdm import tqdm 
import json
from typing import Dict, Set, Any

def get_ngram_frequency(data: Dict[str, Any], test_ngrams: Set[str], args: argparse.Namespace) -> Dict[str, str]: 
    """
    Compute the n-gram frequencies for a given data point.

    Args:
        data (Dict[str, Any]): The data point.
        test_ngrams (Set[str]): The n-grams to compute the frequencies for.
        args (argparse.Namespace): The command-line arguments.

    Returns:
        Dict[str, str]: The n-gram frequencies for the data point.
    """

    # For one datapoint, we want to compute the ngrams that occur in the text and output
    # A dict with the ngram as key and the frequency as value

    data_ngrams = ngrams(data[args.column].split(), args.n)
    data_ngrams = [' '.join(ngram) for ngram in data_ngrams]
    data_ngrams = set(data_ngrams)

    # Compute the intersection between the ngrams of the data and the ngrams of the test set
    intersection = data_ngrams.intersection(test_ngrams)

    # Compute the frequency of the intersection ngrams
    ngram_frequency = {ngram: 0 for ngram in intersection}
    for ngram in intersection:
        ngram_frequency[ngram] = data[args.column].count(ngram) # Count the number of times the ngram occurs in the text

    return {
        'text': json.dumps(ngram_frequency),
    }

def main(args):
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)

    # Load test ngrams
    with jsonlines.open(args.test_ngrams) as reader:
        test_ngrams = [obj['ngram'] for obj in reader]

    test_ngrams = set(test_ngrams)

    # Use map to compute the ngram frequency for each datapoint
    freq_ds = ds.map(lambda x: get_ngram_frequency(x, test_ngrams, args), batched=False, num_proc=args.n_processes)

    # Aggregate the ngram frequencies
    ngram_frequency = {ngram: 0 for ngram in test_ngrams}
    for obj in tqdm(freq_ds['train']):
        obj_ngram_freqs = json.loads(obj['text'])
        for ngram in obj_ngram_freqs:
            ngram_frequency[ngram] += obj_ngram_freqs[ngram]

    ngram_frequency = [{'ngram': ngram, 'frequency': ngram_frequency[ngram]} for ngram in ngram_frequency]

    # Sort it by frequency (low to high)
    ngram_frequency = sorted(ngram_frequency, key=lambda x: x['frequency'])
    
    # Save it
    with jsonlines.open(args.output, 'w') as writer:
        writer.write_all(ngram_frequency)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='keirp/open-web-math-hq-dev')
    parser.add_argument('--column', type=str, default='text')
    parser.add_argument('--test_ngrams', type=str, default='data/test_ngrams.jsonl')
    parser.add_argument('--n', type=int, default=13)
    parser.add_argument('--n_processes', type=int, default=8)
    parser.add_argument('--output', type=str, default='data/test_ngram_frequencies.jsonl')
    parser.add_argument('--cache_dir', type=str)

    args = parser.parse_args()
    main(args)