import argparse
from datasets import load_dataset
from nltk.util import ngrams
import jsonlines
from typing import Dict, List, Tuple, Set

def split_on_ngrams(text: str, ngrams: Set[str], padding: int = 0) -> List[str]:
    """
    Split the given text on the specified n-grams.

    Args:
        text (str): The text to split.
        ngrams (Set[str]): The n-grams to split the text on.
        padding (int, optional): The number of characters to remove before and after each n-gram in the split. Defaults to 0.

    Returns:
        List[str]: The text splits.
    """

    # Find all ngram occurrences and their positions
    ngram_positions = []
    for ngram in ngrams:
        start = 0
        while start < len(text):
            start = text.find(ngram, start)
            if start == -1:
                break
            ngram_positions.append((max(0, start-padding), min(len(text), start+len(ngram)+padding)))
            start += len(ngram)  # move past this ngram

    # Sort the occurrences by start position
    ngram_positions.sort()

    # Merge overlapping occurrences
    merged_ngram_positions = []
    for start, end in ngram_positions:
        if merged_ngram_positions and start <= merged_ngram_positions[-1][1]:
            # This ngram overlaps with the previous one, so merge them
            prev_start, prev_end = merged_ngram_positions.pop()
            merged_ngram_positions.append((prev_start, max(end, prev_end)))
        else:
            # This ngram doesn't overlap with the previous one, so add it as is
            merged_ngram_positions.append((start, end))

    # Now you can split the text using these merged positions
    splits = []
    prev_end = 0
    for start, end in merged_ngram_positions:
        splits.append(text[prev_end:start])  # Only include the text before the ngram
        prev_end = end
    splits.append(text[prev_end:])

    # Remove empty strings from the list
    splits = [s for s in splits if s]

    return splits

def split_on_contamination(data_batch: Dict[str, List[str]], test_ngrams: Set[str], args: argparse.Namespace) -> Dict[str, List[str]]:
    """
    Split the data in the batch on the specified n-grams.

    Args:
        data_batch (Dict[str, List[str]]): The data batch to split.
        test_ngrams (Set[str]): The n-grams to split the data on.
        args (argparse.Namespace): The command-line arguments.

    Returns:
        Dict[str, List[str]]: The split data batch.
    """
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
    for i in range(len(data_batch[args.column])):
        intersection = ngrams_batch[i].intersection(test_ngrams)
        if intersection:            
            # Split the text on the ngrams
            split_text = split_on_ngrams(data_batch[args.column][i], intersection, padding=args.split_padding)

            if len(split_text) > args.max_splits:
                # If the document is split into too many pieces, just remove the document.
                continue

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
    with jsonlines.open(args.test_ngrams) as reader:
        test_ngrams = [obj for obj in reader]

    # Filter out ngrams with frequency below threshold
    test_ngrams = [obj['ngram'] for obj in test_ngrams if obj['frequency'] <= args.ngram_frequency_threshold and obj['frequency'] > 0]

    print('Decontaminating dataset')
    decontaminated_dataset = ds.map(lambda x: split_on_contamination(x, test_ngrams, args), batched=True, batch_size=1000, num_proc=args.n_processes)

    print(decontaminated_dataset)

    if args.save_to_disk:
        print('Saving to disk...')
        decontaminated_dataset.save_to_disk(args.save_to_disk)
    if args.push_to_hub:
        print('Pushing to hub...')
        decontaminated_dataset.push_to_hub(args.push_to_hub)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='keirp/open-web-math-hq-dev')
    parser.add_argument('--column', type=str, default='text')
    parser.add_argument('--test_ngrams', type=str, default='data/test_ngram_frequencies.jsonl')
    parser.add_argument('--ngram_frequency_threshold', type=int, default=10)
    parser.add_argument('--max_splits', type=int, default=10)
    parser.add_argument('--split_padding', type=int, default=200)
    parser.add_argument('--n', type=int, default=13)
    parser.add_argument('--n_processes', type=int, default=8)
    parser.add_argument('--push_to_hub', type=str, default=None)
    parser.add_argument('--save_to_disk', type=str, default=None)
    parser.add_argument('--cache_dir', type=str)

    args = parser.parse_args()
    main(args)