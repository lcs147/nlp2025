# %%
from common import * # import all file paths and common functions and variables

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from tqdm import tqdm
import asyncio
import pytest, json
import torch
import sys


print("Before sys.path:", sys.path)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, os.pardir)
sys.path.append(parent_dir)
print("Current sys.path:", sys.path)
from custom_metrics.execution_accuracy import ExecutionAccuracy
print("sys.path after adding '..':", sys.path)
sys.path.append("./scripts")


# %%
final_lora_paths = [
    f'{CHECKPOINT_LORAS}/{BASE_MODEL_NAME}-lora/checkpoint-261', # 3º época
    f'{CHECKPOINT_LORAS}/{BASE_MODEL_NAME}-lora/checkpoint-435', # 5º época
]

# %%
import os

@pytest.mark.asyncio
async def test_model_on_spider(model_path):
    # read dataset
    print("Loading Spider dev dataset...")
    with open(PATH_SPIDER_DEV_PROCESSED) as f:
        spider_dev_data = json.load(f)
    spider_dev_data = spider_dev_data[:EVALUATION_ENTRIES_SIZE]

    # Checkpoint file path
    checkpoint_file = f"{CHECKPOINTS_DIR}/{'_'.join(model_path.split('/')[-2:])}_testcases_checkpoint.json"

    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            test_cases_data = json.load(f)
        test_cases_data = test_cases_data[:EVALUATION_ENTRIES_SIZE]
        start_idx = len(test_cases_data)
        print(f"Resuming from checkpoint: {start_idx} test cases loaded.")
    else:
        test_cases_data = []
        start_idx = 0

    # load model
    print(f"Loading model from {model_path}...")
    tokenizer, model = custom_load_model(model_path, goal="inference")

    # generate test cases
    test_cases = []
    for i, entry in enumerate(tqdm(spider_dev_data, desc="Generating test cases")):
        if i < start_idx:
            # Already processed
            continue

        prompt = spider_to_sql_inference_prompt(entry)
        generated_sql = execute_prompt(model, tokenizer, prompt)

        case = {
            "input": entry["question"],
            "actual_output": generated_sql,
            "expected_output": entry["query"],
            "context": [entry["db_id"]]
        }
        test_cases_data.append(case)

        # Save checkpoint after each test case
        if (i + 1) % 10 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(test_cases_data, f)

    # Convert to LLMTestCase objects
    for case in test_cases_data:
        test_cases.append(
            LLMTestCase(
                input=case["input"],
                actual_output=case["actual_output"],
                expected_output=case["expected_output"],
                context=case["context"]
            )
        )

    # Define the metric
    execution_accuracy_metric = ExecutionAccuracy(threshold=1)

    # Evaluate
    print("Evaluating test cases...")
    return evaluate(test_cases, [execution_accuracy_metric])

import statistics
# %%
async def main():
    accuracies = []

    for model_path in final_lora_paths:
        dirs = model_path.split('/')
        checkpoint_name = dirs[-1]
        lora_name = dirs[-2]

        print(f"Testing lora: {lora_name}/{checkpoint_name}")
        result = await test_model_on_spider(model_path) # Await here
        accuracies.append((model_path, result))

    for model_path, result in accuracies:
        dirs = model_path.split('/')
        checkpoint_name = dirs[-1]
        lora_name = dirs[-2]

        scores = [m.score for t in result.test_results for m in t.metrics_data]

        accuracy = sum(scores) / len(scores) if scores else 0
        std_dev = statistics.stdev(scores)

        print(f"Lora: {lora_name}/{checkpoint_name} - Accuracy: {accuracy:.3f} ± {std_dev:.3f} ({len(scores)} examples)")

if __name__ == "__main__":
    asyncio.run(main()) # Run the async main function

