#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import evaluate
import torch
import ast

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import Dataset
from collections import Counter

# ------------------------------
# Data Loading & Label Parsing
# ------------------------------

# Load CSV file into a DataFrame
data = pd.read_csv('/Users/zifanli/PycharmProjects/Thematic investing NLP/.venv/SEntFiN-v1.1.csv')
df = pd.DataFrame(data)

# Parse and extract sentiment from label strings like "{'AAPL': 'Positive'}"
def extract_sentiment(label_str):
    try:
        label_dict = ast.literal_eval(label_str)
        # The key (e.g., 'AAPL') is ignored and the value is taken as the sentiment.
        sentiment = list(label_dict.values())[0]
        return sentiment
    except Exception as e:
        print(f"Error parsing label {label_str}: {e}")
        return None

# Rename column and extract sentiment
df = df.rename(columns={'Decisions': 'label'})
df['label'] = df['label'].apply(extract_sentiment)

# Print unique labels for verification
unique_labels = sorted(df['label'].unique())
print("Unique labels:", unique_labels)

# Map sentiment labels (e.g., "Positive", "Negative", etc.) to integer IDs
label2id = {label: idx for idx, label in enumerate(unique_labels)}
df['label'] = df['label'].map(label2id)
print("Label mapping:", label2id)

# ------------------------------
# Tokenizer & Sentiment Model Setup
# ------------------------------

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
language_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(unique_labels)
)
print(f"Model's token dictionary size: {language_model.config.vocab_size}")

# ------------------------------
# Data Splitting
# ------------------------------

# Split into training and test sets
test_df = df.sample(frac=0.2, random_state=731)
train_df = df.drop(test_df.index)

# Further split train into training and validation sets
valid_df = train_df.sample(frac=0.2, random_state=731)
new_train_df = train_df.drop(valid_df.index)

# Convert DataFrames to Hugging Face Datasets
dataset_train = Dataset.from_pandas(new_train_df)
dataset_valid = Dataset.from_pandas(valid_df)
dataset_test  = Dataset.from_pandas(test_df)

# ------------------------------
# Tokenization (with label preservation)
# ------------------------------

def tokenize_function(examples):
    # Tokenize the 'Title' field with a fixed max_length
    tokens = tokenizer(
        examples["Title"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokens["label"] = examples["label"]
    return tokens

# Apply tokenization to datasets (batched)
train_tokenized = dataset_train.map(tokenize_function, batched=True)
valid_tokenized = dataset_valid.map(tokenize_function, batched=True)
test_tokenized  = dataset_test.map(tokenize_function, batched=True)

# Remove any unnecessary columns; keep only input_ids, attention_mask, and label
cols_to_keep = ["input_ids", "attention_mask", "label"]
train_tokenized = train_tokenized.remove_columns([col for col in train_tokenized.column_names if col not in cols_to_keep])
valid_tokenized = valid_tokenized.remove_columns([col for col in valid_tokenized.column_names if col not in cols_to_keep])
test_tokenized  = test_tokenized.remove_columns([col for col in test_tokenized.column_names if col not in cols_to_keep])

# ------------------------------
# Setup Device
# ------------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
language_model.to(device)

# ------------------------------
# Define Metrics & Training Arguments
# ------------------------------

metric = evaluate.load("accuracy")

#%% No need to run this if you already download the fine-tuned model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="steps",  # Evaluate every few steps
    eval_steps=50,
    logging_steps=50,
    save_steps=500,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=language_model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    compute_metrics=compute_metrics
)

# ------------------------------
# Train the Sentiment Model
# ------------------------------

trainer.train()

#%%

# Save the sentiment model and tokenizer
trainer.save_model("./my_fine_tuned_model")
tokenizer.save_pretrained("./my_fine_tuned_model")


#%%
# ------------------------------
# Zero-Shot Sector Classification Setup
# ------------------------------

candidate_sectors = [
    "Financials",
    "Information Technology",
    "Consumer Discretionary",
    "Healthcare",
    "Utilities",
    "Industrials",
    "Energy",
    "Materials",
    "Real Estate",
    "Communication Services",
    "Miscellaneous"
]

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ------------------------------
# Named Entity Recognition (NER) Setup
# ------------------------------

# We use a pre-trained NER pipeline to extract organizations (which may represent securities/companies)
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"
)

# ------------------------------
# Inference: Sentiment, Sector, and Securities (NER) Extraction
# ------------------------------

# Reload model and tokenizer (if needed)
model = AutoModelForSequenceClassification.from_pretrained("./my_fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./my_fine_tuned_model")
model.eval()  # Set to evaluation mode
model.to(device)

# Create a reverse mapping from label id to sentiment string
id2label = {v: k for k, v in label2id.items()}

def predict_sentiment(text: str) -> str:
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = int(torch.argmax(logits, dim=1).item())
    sentiment = id2label.get(predicted_class_id, "Unknown")
    return sentiment

def predict_sector(text: str) -> str:
    result = zero_shot_classifier(text, candidate_sectors)
    return result["labels"][0]

def extract_securities(text: str) -> list:
    # Use the NER pipeline to extract entities from text
    # We filter for organizations (entity_group == "ORG")
    ner_results = ner_pipeline(text)
    securities = [ent["word"] for ent in ner_results if ent["entity_group"] == "ORG"]
    return securities

# Loop for user input and predictions
while True:
    text = input("Enter a sentence for sentiment, sector, and securities extraction (or type 'quit' to exit): ")
    if text.lower() == "quit":
        break
    sentiment_prediction = predict_sentiment(text)
    sector_prediction = predict_sector(text)
    securities_extracted = extract_securities(text)
    print(f"Predicted sentiment: {sentiment_prediction}")
    print(f"Predicted sector: {sector_prediction}")
    print(f"Extracted securities/organizations: {securities_extracted if securities_extracted else 'None found'}")

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Extract log history from the Trainer state (assumes 'trainer' is your Trainer instance)
log_history = trainer.state.log_history
df_logs = pd.DataFrame(log_history)

# Filter rows: training logs contain "loss", evaluation logs contain "eval_loss"
train_loss_df = df_logs[df_logs["loss"].notnull()]
eval_loss_df = df_logs[df_logs["eval_loss"].notnull()]

plt.figure(figsize=(10, 6))
plt.plot(train_loss_df["step"], train_loss_df["loss"], label="Training Loss", marker='o', linestyle='-')
plt.plot(eval_loss_df["step"], eval_loss_df["eval_loss"], label="Validation Loss", marker='x', linestyle='--')
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs. Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
