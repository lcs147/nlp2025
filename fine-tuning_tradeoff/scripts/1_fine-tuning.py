# %%
from common import * # import all file paths and common functions and variables

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
import torch, flash_attn
import json, random
import numpy as np

# %%
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"CUDA compute capability: {major}.{minor}")
    if major >= 8:
        print("Your GPU supports bf16 (bfloat16) and flash_attention_2.")
    else:
        print("Your GPU does NOT support bf16 (bfloat16).")
else:
    print("CUDA is not available.")

# %%
# Load processed Spider dataset
with open(PATH_SPIDER_DEV_PROCESSED) as f:
    train_data = json.load(f)

# with open(PATH_SPIDER_TEST_PROCESSED) as f:
#     eval_data = json.load(f)
# eval_data = random.sample(eval_data, 100)

# %%
print(f"Number of lines in the train dataset: {len(train_data)}")
# print(f"Number of lines in the eval dataset: {len(eval_data)}")

# %%
# Converting entries to the Spider to SQL prompt format

# Build the dataset for supervised fine-tuning from train_data
supervised_train_data = []
for entry in train_data:
    prompt = spider_to_sql_full_prompt(entry)
    supervised_train_data.append({"text": prompt})

# Build the dataset for supervised fine-tuning from eval_data
# supervised_eval_data = []
# for entry in eval_data:
#     prompt = spider_to_sql_full_prompt(entry)
#     supervised_eval_data.append({"text": prompt})


# %%
train_dataset = Dataset.from_list(supervised_train_data)
# val_dataset = Dataset.from_list(supervised_eval_data)

# %%
from transformers import BitsAndBytesConfig

def train_model(model_id, name_tag):
    output_name = f"{model_id.split('/')[-1]}-{name_tag}"
    output_dir = f"{CHECKPOINT_LORAS}/{output_name}"
    
    print(f"\n=== Training {output_name} ===\n")

    # load the model and tokenizer
    tokenizer, model = custom_load_model(model_id, goal="training")
    print(model.eval())
    print(model.device)

    # tokenize the dataset
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    # tokenized_val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        group_by_length=True,
        output_dir=output_dir,
        save_strategy="epoch",
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        seed=SEED,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        # eval_dataset=tokenized_val_dataset,
        data_collator=data_collator
    )

    try:
        print("Attempting to resume training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        print("Failed to resume training from checkpoint.")
        print(e)
        print("Starting training from scratch...")
        trainer.train()
    
    # Save the trained model
    model.save_pretrained(output_dir)
    print(f"Finished training {output_name}.")

# %%
train_model(model_id=BASE_MODEL_ID, name_tag="lora")


