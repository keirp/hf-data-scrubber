# hf-data-scrubber

## Creating a test set

This repo contains a simple script that will create a test set for decontamination containing the test splits of the following benchmarks:

- gsm8k
- MMLU
- Hendrycks MATH
- ProofNet
- MiniF2F

To run this script, run:

```python create_test_set.py```

## Decontaminating a HuggingFace dataset

This Python script can be used to decontaminate a HuggingFace dataset. The dataset is analyzed for n-grams (default is 13-grams) that match those found in a test set. Contaminating n-grams are then removed from the dataset, splitting the text at these n-grams.

Usage
To use the script, run:

```python hf_data_scrubber.py --dataset <dataset> --column <column> --test_set <test_set> --n <n> --n_processes <n_processes>```

Where:

- `dataset` is the HuggingFace dataset to be decontaminated (default is 'keirp/open-web-math-hq-dev')
- `column` is the column to be analyzed for contamination (default is 'text')
- `test_set` is the .jsonl file containing the test set (default is 'data/test.jsonl')
- `n` is the number of words in the n-grams (default is 13)
- `n_processes` is the number of processes for batch processing (default is 8)