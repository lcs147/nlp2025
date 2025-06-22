# %%
from common import * # import all file paths and common functions and variables

# %%
from datasets import load_dataset
import random
import json

# %% [markdown]
# # Preparando o Spider

# %%
def load_spider_dataset(file_name):
    """Loads the Spider dataset from JSON files."""
    with open(PATH_SPIDER_FULL + '/' +  file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(PATH_SPIDER_FULL + '/tables.json', 'r', encoding='utf-8') as f:
        tables = json.load(f)

    # Create a mapping from db_id to table schema
    db_schemas = {table['db_id']: table for table in tables}

    processed_data = []
    for item in data:
        db_id = item['db_id']
        question = item['question']
        query = item['query']
        schema = db_schemas.get(db_id, {})

        # list tables and columns.
        db_schema_text = f"Database: {db_id}\n"
        if 'table_names_original' in schema and 'column_names_original' in schema:
            for i, table_name in enumerate(schema['table_names_original']):
                db_schema_text += f"Table {table_name} ("
                cols = [col[1] for col in schema['column_names_original'] if col[0] == i]
                db_schema_text += ", ".join(cols)
                db_schema_text += ")\n"

        processed_data.append({
            "db_id": db_id,
            "question": question,
            "query": query,
            "db_schema": db_schema_text
        })
    return processed_data

# %%
print("Processing dev.json...")
processed_data = load_spider_dataset('dev.json')
print(processed_data[:2])

with open(PATH_SPIDER_DEV_PROCESSED, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

# %%
print("Processing test.json...")
processed_data = load_spider_dataset('test.json')
print(processed_data[:5])
with open(PATH_SPIDER_TEST_PROCESSED, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

# %% [markdown]
# # Preparando o MMLU

# %%
subcategorias = [
    ("elementary_mathematics", "STEM"),
    ("philosophy", "Humanidades"),
    ("management", "Ciências Sociais")
]
n_perguntas_subcat = 50

# %%
mmlu_questions = []
for subcategoria, categoria in subcategorias:
    dataset = load_dataset("cais/mmlu", subcategoria, split="test")
    
    shuffled_dataset = dataset.shuffle(seed=SEED) 
    sampled_questions = shuffled_dataset.select(range(n_perguntas_subcat))
    
    # add the category to each question
    sampled_questions = sampled_questions.map(lambda q: {**q, 'category': categoria})
    
    mmlu_questions.extend(sampled_questions)

random.seed(SEED)
random.shuffle(mmlu_questions)
random.shuffle(mmlu_questions)

# %%
print(mmlu_questions)

# %%
from collections import Counter
subject_counts = Counter(item['subject'] for item in mmlu_questions)
print(subject_counts)

# %%
with open(PATH_MMLU_PROCESSED, "w", encoding="utf-8") as f:
    json.dump(mmlu_questions, f, ensure_ascii=False, indent=2)


