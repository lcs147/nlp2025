# %%
from common import * # import all file paths and common functions and variables

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json, sqlite3
import torch

# 3 exemplos do train dataset
few_shot_entries = [
  {
    "db_id": "department_management",
    "question": "How many heads of the departments are older than 56 ?",
    "query": "SELECT count(*) FROM head WHERE age > 56",
    "db_schema": "Database: department_management\nTable department (Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees)\nTable head (head_ID, name, born_state, age)"
  },
  {
    "db_id": "bike_1",
    "question": "What is the name and country of the manufacturer with id 2?",
    "query": "SELECT Name , Country FROM manufacturer WHERE Manufacturer_ID = 2",
    "db_schema": "Database: bike_1\nTable manufacturer (Manufacturer_ID, Name, Country)"
  },
  {
    "db_id": "station_weather",
    "question": "What is the minimum, maximum and average temperature?",
    "query": "SELECT min(T_min) , max(T_max) , avg(T_min) FROM station",
    "db_schema": "Database: station_weather\nTable station (T_min, T_max)"
  }
]

few_shot_prompts = [spider_to_sql_full_prompt(entry) for entry in few_shot_entries]
few_shot_examples = "\n".join(few_shot_prompts)
print(few_shot_examples)


# %%
# Implement a simple SQLite execution
def execute_sql(db_path, query):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        return True, cursor.fetchall()
    except sqlite3.Error as e:
        # print(f"SQL Error: {e} for query: {query}")
        return False, str(e)
    finally:
        if conn:
            conn.close()

# %%
tokenizer, model = custom_load_model(BASE_MODEL_ID, goal="inference")

print(model.eval())
print(model.device)

# %%
with open(PATH_SPIDER_DEV_PROCESSED, "r") as f:
    spider_dev_data = json.load(f)
print(f"Loaded {len(spider_dev_data)} entries from {PATH_SPIDER_DEV_PROCESSED}")

# %%
import os
import json
import re

# Try to load checkpoint
if os.path.exists(CHECKPOINT_BASELINE):
    with open(CHECKPOINT_BASELINE, "r") as f:
        checkpoint = json.load(f)
    results = checkpoint["results"]
    start_idx = checkpoint["last_index"] + 1
    print(f"Resuming from index {start_idx}")
else:
    results = []
    start_idx = 0

for i, entry in enumerate(spider_dev_data[start_idx:EVALUATION_ENTRIES_SIZE], start=start_idx):
    db_path = f"{PATH_SPIDER_FULL}/database/{entry['db_id']}/{entry['db_id']}.sqlite"
    
    prompt = spider_to_sql_inference_prompt(entry)
    prompt = few_shot_examples + "\n" + prompt
    
    # execute prompt
    extracted_sql = execute_prompt(model, tokenizer, prompt)

    # execute the sql queries
    generated_success, generated_result = execute_sql(db_path, extracted_sql)
    ground_truth_success, ground_truth_result = execute_sql(db_path, entry['query'])

    print(f"Generated Result: {generated_result}")
    print(f"Ground Truth Result: {ground_truth_result}")

    is_correct = False
    if generated_success and ground_truth_success:
        is_correct = set(generated_result) == set(ground_truth_result)
    
    results.append({
        "question": entry['question'],
        "ground_truth_sql": entry['query'],
        "generated_sql": extracted_sql,
        "is_correct_baseline": is_correct
    })
    print(f"Processed {i+1}/{len(spider_dev_data)}: Success={is_correct}")

    if (i + 1) % 10 == 0:
        print(f"Checkpointing at index {i + 1}")
        with open(CHECKPOINT_BASELINE, "w") as f:
            json.dump({"results": results, "last_index": i}, f)

results = results[:EVALUATION_ENTRIES_SIZE]
# %%

import statistics
ncorrect = sum(1 for r in results if r.get("is_correct_baseline"))
accuracy = ncorrect / len(results)

correctness_values = [1 if r.get("is_correct_baseline") else 0 for r in results]
std_dev = statistics.stdev(correctness_values)

print(f"Accuracy: {accuracy:.3f} ± {std_dev:.3f} ({len(results)} examples)")


