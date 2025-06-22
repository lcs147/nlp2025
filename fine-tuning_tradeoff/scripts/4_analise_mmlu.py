# %%
from common import * # import all file paths and common functions and variables

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import json
import re

# %%
with open(PATH_MMLU_PROCESSED) as f:
    mmlu_data = json.load(f)

# %%
mmlu_few_shot_examples = [
    {
        "question": "Which data structure is most efficient for implementing a call stack in a programming language?",
        "choices": ["A. Queue", "B. Stack", "C. Heap", "D. Tree"],
        "answer": 1
    },
    {
        "question": "Which philosopher is known for the phrase 'I think, therefore I am'?",
        "choices": ["A. John Locke", "B. René Descartes", "C. Immanuel Kant", "D. David Hume"],
        "answer": 1
    },
    {
        "question": "Which tool is commonly used by central banks to influence the money supply?",
        "choices": ["A. Open market operations", "B. Taxation", "C. Wage controls", "D. Price ceilings"],
        "answer": 0
    },
    {
        "question": "Which Roman emperor divided the empire into Eastern and Western regions?",
        "choices": ["A. Augustus", "B. Diocletian", "C. Constantine", "D. Nero"],
        "answer": 1
    }
]

# %%
def create_mmlu_prompt(question_data, few_shot_examples):
    prompt = ""
    for ex in few_shot_examples:
        prompt += (
            f"<|im_start|>user\n"
            f"The following is a multiple choice question. Choose the correct answer.\n"
            f"Question: {ex['question']}\n"
            f"{' '.join(ex['choices'])}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"Answer: {ex['answer']}<|im_end|>\n"
        )

    prompt += (
        f"<|im_start|>user\n"
        f"The following is a multiple choice question. Choose the correct answer.\n"
        f"Question: {question_data['question']}\n"
        f"{' '.join(question_data['choices'])}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"Answer: "
    )
    return prompt

# %%
from tqdm import tqdm

def evaluate_mmlu(model_path):
    tokenizer, model = custom_load_model(model_path, goal="inference")

    total_correct = 0
    category_correct = {"STEM": 0, "Humanities": 0, "Social Sciences": 0}
    results = []

    for entry in tqdm(mmlu_data, desc="Answering MMLU questions with model"):
        category = entry['category']

        prompt = create_mmlu_prompt(entry, mmlu_few_shot_examples)
        generated_text = execute_prompt(model, tokenizer, prompt)

        predicted_answer = ""
        match = re.search(r"^[0-3]", generated_text) # Look for 0, 1, 2, or 3 at the start
        predicted_answer = match.group(0) if match else "@"

        is_correct = (predicted_answer.isdigit() and int(predicted_answer) == entry["answer"])

        results.append({
            "question": entry["question"],
            "category": category,
            "choices": entry["choices"],
            "correct_answer": entry["answer"],
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })
        # print(f"MMLU Q: Correct: {entry["answer"]} | Predicted: {predicted_answer} | Is Correct: {is_correct}")

    return results

# %%
models_list = [
    BASE_MODEL_ID,
    f"{CHECKPOINT_LORAS}/{BASE_MODEL_NAME}-lora/checkpoint-261", # 3 epoca
    f"{CHECKPOINT_LORAS}/{BASE_MODEL_NAME}-lora/checkpoint-435", # 5 epoca
]

# %%
import os
import json
from collections import defaultdict

for model_path in models_list:
    model_name = '_'.join(model_path.split('/')[-2:])
    results_file = f"{CHECKPOINTS_DIR}/{model_name.replace('/', '_')}_mmlu_results.json"

    # Do evaluation only if it was not done before
    if os.path.exists(results_file):
        print(f"Evaluation for {model_name} is already done. Skipping evaluation.")
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        print(f"Evaluating model: {model_name}")
        results = evaluate_mmlu(model_path)
        with open(results_file, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


# %%
import numpy as np

for model_path in models_list:
    model_name = '_'.join(model_path.split('/')[-2:])
    results_file = f"{CHECKPOINTS_DIR}/{model_name.replace('/', '_')}_mmlu_results.json"

    if os.path.exists(results_file):
        print(f"Results for {model_name}.")
        with open(results_file, "r") as f:
            results = json.load(f)

    # Calculate overall accuracy
    total = len(results)
    total_correct = sum(r["is_correct"] for r in results)
    accuracy = total_correct / total if total > 0 else 0

    # Accuracy by category
    category_correct = defaultdict(int)
    category_accs = defaultdict(list)
    for r in results:
        cat = r["category"]
        if r["is_correct"]:
            category_correct[cat] += 1
        category_accs[cat].append(int(r["is_correct"]))

    categories = category_correct.keys()
    cat_accuracies = [
        category_correct[cat] / 50 for cat in categories
    ]
    std_acc = np.std(cat_accuracies)

    print(f"Aggregated accuracy: {accuracy:.3f} ± {std_acc:.3f}")
    print("Accuracy by category:")
    for cat in categories:
        acc = category_correct[cat] / 50
        cat_std = np.std(category_accs[cat])
        print(f"  {cat}: {acc:.3f} ± {cat_std:.3f}")
    print("-" * 40)



