import argparse
from nltk.util import ngrams
import jsonlines
from tqdm import tqdm

def main(args):
    # Load the test set
    with jsonlines.open(args.test_set) as reader:
        test_set = [obj for obj in reader]

    # Compute 13-grams for the test set
    test_ngrams = []
    for obj in tqdm(test_set):
        obj_ngrams = ngrams(obj.split(), args.n)
        obj_ngrams = [' '.join(ngram) for ngram in obj_ngrams]
        test_ngrams.extend(list(obj_ngrams))
    test_ngrams = set(test_ngrams)

    # Format it like {'ngram': _, 'frequency': None}
    test_ngrams = [{'ngram': ngram, 'frequency': None} for ngram in test_ngrams]

    # Save it
    with jsonlines.open(args.output, 'w') as writer:
        writer.write_all(test_ngrams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='data/test.jsonl')
    parser.add_argument('--n', type=int, default=13)
    parser.add_argument('--output', type=str, default='data/test_ngrams.jsonl')

    args = parser.parse_args()
    main(args)