"""Create a test set from popular LLM evaluation datasets."""
import argparse
import inspect
from datasets import load_dataset
import jsonlines
import custom_datasets.hendrycks_math.hendrycks_math
import glob

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

MATH_SUBJECTS = [
    'algebra', 
    'counting_and_probability', 
    'geometry', 
    'intermediate_algebra', 
    'number_theory', 
    'prealgebra', 
    'precalculus'
]

def get_texts(path):
    # Use glob
    texts = []
    for file in glob.glob(path):
        with open(file, 'r') as f:
            texts.append(f.read())
    return texts

def format_mmlu(doc):
    """We only deduplicate against the question text since finding the choices in the exact order
    seems unlikely."""
    question = doc["question"].strip()
    return question

def add_texts_to_output(texts, output):
    with jsonlines.open(output, mode='a') as writer:
        for txt in texts:
            writer.write(txt)

def main(args):
    # Remove existing test set
    with jsonlines.open(args.output, mode='w') as writer:
        writer.write_all([])

    # Add gsm8k test set
    ds = load_dataset('gsm8k', 'main', split='test')
    add_texts_to_output([obj['answer'] for obj in ds], args.output)

    # Add MMLU test set
    for subject in MMLU_SUBJECTS:
        ds = load_dataset('cais/mmlu', subject, split='test')
        add_texts_to_output([format_mmlu(obj) for obj in ds], args.output)

    # Add Hendrycks MATH test set
    for subject in MATH_SUBJECTS:
        ds = load_dataset(inspect.getfile(custom_datasets.hendrycks_math.hendrycks_math), subject, split='test')
        # Add all problems
        add_texts_to_output([obj['problem'] for obj in ds], args.output)
        # Add all solutions
        add_texts_to_output([obj['solution'] for obj in ds], args.output)

    # Add ProofNet
    ds = load_dataset('hoskinson-center/proofnet', split='test')
    # Add nl_statement
    add_texts_to_output([obj['nl_statement'] for obj in ds], args.output)
    # Add nl_proof
    add_texts_to_output([obj['nl_proof'] for obj in ds], args.output)
    # Add formal_statement
    add_texts_to_output([obj['formal_statement'] for obj in ds], args.output)

    # Add miniF2F
    # Add hollight
    texts = get_texts('miniF2F/hollight/test/*.ml')
    add_texts_to_output(texts, args.output)
    # Add Isabelle
    texts = get_texts('miniF2F/isabelle/test/*.thy')
    add_texts_to_output(texts, args.output)
    # Add Lean
    texts = get_texts('miniF2F/lean/src/test.lean')
    add_texts_to_output(texts, args.output)
    # Add MetaMath
    texts = get_texts('miniF2F/metamath/test/*.mm')
    add_texts_to_output(texts, args.output)

    # Add OCW
    texts = get_texts('custom_datasets/ocw/*.tex')
    add_texts_to_output(texts, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/test.jsonl')
    args = parser.parse_args()
    main(args)