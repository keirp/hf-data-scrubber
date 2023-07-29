# hf-data-scrubber :sponge:

## Creating a test set

This repo contains a simple script that will create a test set for decontamination containing the test splits of the following benchmarks:

- gsm8k
- MMLU
- Hendrycks MATH
- ProofNet
- MiniF2F
- OCW

To run this script, run:

```python create_test_set.py```

If you want to decontaminate against a different test set, make sure your test jsonl is formatted as a jsonl where each line is a string.

### Example:

```json
"Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
"It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"
"The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"
```

## Decontaminating a HuggingFace dataset

This repository follows the methodology of the GPT-3 paper found [here in Appendix C](https://arxiv.org/abs/2005.14165). At a high level, the process involves:

1. Get the ngrams from the test set
2. Compute how frequently they occur in the training set
3. Remove the ngrams that occur more than a threshold (default is 10) times in the training set
4. For each document, remove the ngrams and 200 characters before and after the ngram, splitting the document. If the document is split into more than 10 parts, remove the document.
5. Save the decontaminated dataset.


### Usage

To compute the test set ngrams, run:

```bash
python create_test_ngrams.py --test_set <path to test set> --n <ngram size> --output <path to output file>
```

To compute the ngram frequencies, run:

```bash
python compute_ngram_frequencies.py --dataset <huggingface dataset name> --column <column to decontaminate> --test_ngrams <path to test ngrams> --n <ngram size> --n_processes <number of processes to use> --output <path to output file>
```

Finally, to decontaminate the dataset, run:

```bash
python decontaminate.py --dataset <huggingface dataset name> --column <column to decontaminate> --test_ngrams <path to ngram frequencies> --n <ngram size> --ngram_frequency_threshold <ngram frequency threshold> --max_splits <max splits> --split_padding <split padding> --n_processes <number of processes to use> --push_to_hub <name for huggingface hub> --save_to_disk <path to save decontaminated dataset>
```
