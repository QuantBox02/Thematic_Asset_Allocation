#%%

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm.auto import tqdm

#%%
# Enable progress_apply for Pandas
tqdm.pandas()

# ------------------------------
# Load the Fine-Tuned Sentiment Model
# ------------------------------
model_path = "./my_fine_tuned_model"  # Adjust the path if needed
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Use the model's config mapping if available; otherwise, provide a fallback mapping.
if hasattr(model.config, "id2label") and model.config.id2label is not None:
    id2label = model.config.id2label
else:
    id2label = {0: "Negative", 1: "Positive"}

# ------------------------------
# Setup Zero-Shot and NER Pipelines
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
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# ------------------------------
# Define Prediction Functions
# ------------------------------
def predict_sentiment(text: str) -> str:
    """Predict sentiment using the fine-tuned model."""
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = int(torch.argmax(logits, dim=1).item())
    return id2label.get(predicted_class_id, "Unknown")

def predict_sector(text: str) -> str:
    """Predict the sector using zero-shot classification."""
    result = zero_shot_classifier(text, candidate_sectors)
    return result["labels"][0]

def extract_securities(text: str) -> list:
    """Extract organizations (securities) using the NER pipeline."""
    ner_results = ner_pipeline(text)
    securities = [ent["word"] for ent in ner_results if ent["entity_group"] == "ORG"]
    return securities

# ------------------------------
# Load Unlabeled Data and Make Predictions with Progress Bars
# ------------------------------
unlabeled_path = "/Users/zifanli/PycharmProjects/Thematic_Solution/finviz_news.csv"
unlabeled_df = pd.read_csv(unlabeled_path)

# Assuming the scraped data has a "headline" column (you may combine with "article_content" if desired)
unlabeled_df["predicted_sentiment"] = unlabeled_df["headline"].progress_apply(predict_sentiment)
unlabeled_df["predicted_sector"] = unlabeled_df["headline"].progress_apply(predict_sector)
unlabeled_df["predicted_securities"] = unlabeled_df["headline"].progress_apply(extract_securities)

# Save the enriched DataFrame with predictions to a new CSV file
output_path = "/Users/zifanli/PycharmProjects/Thematic_Solution/finviz_news_with_predictions.csv"
unlabeled_df.to_csv(output_path, index=False)

# Optionally, display the first few rows of the enriched DataFrame
print(unlabeled_df.head())

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Set seaborn style for improved aesthetics
sns.set(style="whitegrid", context="talk", palette="muted")

# ------------------------------
# Load the Enriched Data
# ------------------------------
df = pd.read_csv("/Users/zifanli/PycharmProjects/Thematic_Solution/finviz_news_with_predictions.csv")


# ------------------------------
# Map Fine-Tuned Model Labels to Three Classes
# ------------------------------
# Assuming the model returns "LABEL_0", "LABEL_1", "LABEL_2"
def map_label(label):
    mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    return mapping.get(label, "neutral")


df["predicted_sentiment"] = df["predicted_sentiment"].apply(map_label)

# =============================================================================
# 1. Sector-Level Diverging Sentiment Visualization (Using Frequency Density)
# =============================================================================

# Group by sector and sentiment and pivot to get counts.
sector_counts = df.groupby(["predicted_sector", "predicted_sentiment"]).size().reset_index(name="count")
sector_pivot = sector_counts.pivot(index="predicted_sector", columns="predicted_sentiment", values="count").fillna(0)

# Ensure all sentiment columns exist.
for col in ["positive", "negative", "neutral"]:
    if col not in sector_pivot.columns:
        sector_pivot[col] = 0

# Calculate total number of samples per sector.
sector_pivot["total"] = sector_pivot.sum(axis=1)

# Calculate frequency densities (as percentages)
sector_pivot["positive_density"] = sector_pivot["positive"] / sector_pivot["total"] * 100
sector_pivot["negative_density"] = sector_pivot["negative"] / sector_pivot["total"] * 100
sector_pivot["neutral_density"] = sector_pivot["neutral"] / sector_pivot["total"] * 100

# Compute net sentiment as percentage difference between positive and negative.
sector_pivot["net_sentiment"] = (sector_pivot["positive"] - sector_pivot["negative"]) / sector_pivot["total"] * 100

# Rank sectors by net sentiment ascending (for diverging bar chart, lowest on top)
sector_pivot = sector_pivot.sort_values("net_sentiment", ascending=True)

# Prepare data for plotting.
sectors = sector_pivot.index.tolist()
pos_density = sector_pivot["positive_density"]
neg_density = sector_pivot["negative_density"]
total_samples = sector_pivot["total"]
net_sent = sector_pivot["net_sentiment"]
neutral_density = sector_pivot["neutral_density"]

fig, ax = plt.subplots(figsize=(12, 8))
# Plot positive density (to the right)
ax.barh(sectors, pos_density, color="lightgreen", label="Positive")
# Plot negative density (as negative values to the left)
ax.barh(sectors, -neg_density, color="lightcoral", label="Negative")
# Draw vertical line at x=0 for clarity.
ax.axvline(0, color="black", linewidth=1)

# Annotate each bar with net sentiment and total samples, and display neutral density.
for i, sector in enumerate(sectors):
    annotation = (f"Net: {net_sent.loc[sector]:.1f}% | "
                  f"Neutral: {neutral_density.loc[sector]:.1f}% (n={int(total_samples.loc[sector])})")
    ax.text(0, i, annotation,
            va='center', ha='center', color="black", fontsize=12)

ax.set_title("Diverging Sentiment by Sector (Frequency Density)", fontsize=18)
ax.set_xlabel("Frequency (%)", fontsize=16)
ax.set_ylabel("Sector", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14, title="Sentiment", title_fontsize=16)
plt.tight_layout()
plt.show()

# =============================================================================
# 2. Securities-Level Diverging Sentiment Visualization (Using Frequency Density)
# =============================================================================

# Convert the predicted_securities column from string representation to a list.
df["predicted_securities"] = df["predicted_securities"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Explode the securities column so that each security appears on its own row.
df_exploded = df.explode("predicted_securities")

# Count frequency for each security.
security_counts = df_exploded["predicted_securities"].value_counts()
top_n = 10
top_securities = security_counts.head(top_n).index.tolist()

# Filter for top securities.
df_top = df_exploded[df_exploded["predicted_securities"].isin(top_securities)]

# Group by security and sentiment, then pivot.
sec_sent_counts = df_top.groupby(["predicted_securities", "predicted_sentiment"]).size().reset_index(name="count")
security_pivot = sec_sent_counts.pivot(index="predicted_securities", columns="predicted_sentiment",
                                       values="count").fillna(0)

# Ensure all sentiment columns exist.
for col in ["positive", "negative", "neutral"]:
    if col not in security_pivot.columns:
        security_pivot[col] = 0

# Calculate total number of samples per security.
security_pivot["total"] = security_pivot.sum(axis=1)

# Calculate frequency densities (as percentages).
security_pivot["positive_density"] = security_pivot["positive"] / security_pivot["total"] * 100
security_pivot["negative_density"] = security_pivot["negative"] / security_pivot["total"] * 100
security_pivot["neutral_density"] = security_pivot["neutral"] / security_pivot["total"] * 100

# Compute net sentiment as percentage difference.
security_pivot["net_sentiment"] = (security_pivot["positive"] - security_pivot["negative"]) / security_pivot[
    "total"] * 100

# Rank securities by net sentiment ascending.
security_pivot = security_pivot.sort_values("net_sentiment", ascending=True)

# Prepare data for plotting.
securities = security_pivot.index.tolist()
pos_sec_density = security_pivot["positive_density"]
neg_sec_density = security_pivot["negative_density"]
total_sec_samples = security_pivot["total"]
net_sec_sent = security_pivot["net_sentiment"]
neutral_sec_density = security_pivot["neutral_density"]

fig, ax2 = plt.subplots(figsize=(12, 8))
# Plot positive density for securities.
ax2.barh(securities, pos_sec_density, color="lightgreen", label="Positive")
# Plot negative density (as negative values).
ax2.barh(securities, -neg_sec_density, color="lightcoral", label="Negative")
# Draw vertical line at x=0.
ax2.axvline(0, color="black", linewidth=1)

# Annotate each bar with net sentiment and sample count.
for i, sec in enumerate(securities):
    annotation = (f"Net: {net_sec_sent.loc[sec]:.1f}% | "
                  f"Neutral: {neutral_sec_density.loc[sec]:.1f}% (n={int(total_sec_samples.loc[sec])})")
    ax2.text(0, i, annotation,
             va='center', ha='center', color="black", fontsize=12)

ax2.set_title(f"Diverging Sentiment for Top {top_n} Predicted Themes (Frequency Density)", fontsize=18)
ax2.set_xlabel("Sentiment Frequency (%)", fontsize=16)
ax2.set_ylabel("Security", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.legend(fontsize=14, title="Sentiment", title_fontsize=16)
plt.tight_layout()
plt.show()


#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Set seaborn style for improved aesthetics
sns.set(style="whitegrid", context="talk", palette="muted")

# ------------------------------
# Load the Enriched Data
# ------------------------------
df = pd.read_csv("/Users/zifanli/PycharmProjects/Thematic_Solution/finviz_news_with_predictions.csv")

# ------------------------------
# Map Fine-Tuned Model Labels to Three Classes
# ------------------------------
def map_label(label):
    mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    return mapping.get(label, "neutral")

df["predicted_sentiment"] = df["predicted_sentiment"].apply(map_label)

# =============================================================================
# 1. Sector-Level Diverging Sentiment Visualization (Using Frequency Density)
# =============================================================================

# Group by sector and sentiment and pivot to get counts.
sector_counts = df.groupby(["predicted_sector", "predicted_sentiment"]).size().reset_index(name="count")
sector_pivot = sector_counts.pivot(index="predicted_sector", columns="predicted_sentiment", values="count").fillna(0)

# Ensure all sentiment columns exist.
for col in ["positive", "negative", "neutral"]:
    if col not in sector_pivot.columns:
        sector_pivot[col] = 0

# Calculate total number of samples per sector.
sector_pivot["total"] = sector_pivot.sum(axis=1)

# Calculate frequency densities (as percentages).
sector_pivot["positive_density"] = sector_pivot["positive"] / sector_pivot["total"] * 100
sector_pivot["negative_density"] = sector_pivot["negative"] / sector_pivot["total"] * 100
sector_pivot["neutral_density"]  = sector_pivot["neutral"]  / sector_pivot["total"] * 100

# Compute net sentiment as percentage difference.
sector_pivot["net_sentiment"] = (sector_pivot["positive"] - sector_pivot["negative"]) / sector_pivot["total"] * 100

# Rank sectors by net sentiment ascending.
sector_pivot = sector_pivot.sort_values("net_sentiment", ascending=True)

# Prepare data for plotting.
sectors = sector_pivot.index.tolist()
pos_density = sector_pivot["positive_density"]
neg_density = sector_pivot["negative_density"]
total_samples = sector_pivot["total"]
net_sent = sector_pivot["net_sentiment"]
neutral_density = sector_pivot["neutral_density"]

fig, ax = plt.subplots(figsize=(12, 8))
# Plot positive density (to the right)
ax.barh(sectors, pos_density, color="lightgreen", label="Positive")
# Plot negative density (as negative values to the left)
ax.barh(sectors, -neg_density, color="lightcoral", label="Negative")
# Draw vertical line at x=0.
ax.axvline(0, color="black", linewidth=1)

# Define an offset for the annotation (in percentage points).
offset = 2

# Annotate each bar conditionally.
for i, sector in enumerate(sectors):
    ns = net_sent.loc[sector]
    nd = neutral_density.loc[sector]
    count = int(total_samples.loc[sector])
    annotation = f"Net: {ns:.1f}% | Neutral: {nd:.1f}% (n={count})"
    if ns > 0:
        ax.text(-offset, i, annotation, va='center', ha='left', color="Black", fontsize=12)

    elif ns < 0:
        ax.text(offset, i, annotation, va='center', ha='right', color="Black", fontsize=12)
    else:
        ax.text(0, i, annotation, va='center', ha='center', color="Black", fontsize=12)

ax.set_title("Diverging Sentiment by Sector (Frequency Density)", fontsize=18)
ax.set_xlabel("Frequency (%)", fontsize=16)
ax.set_ylabel("Sector", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14, title="Sentiment", title_fontsize=16)
plt.tight_layout()
plt.show()

# =============================================================================
# 2. Securities-Level Diverging Sentiment Visualization (Using Frequency Density)
# =============================================================================

# Convert the predicted_securities column from string representation to a list.
df["predicted_securities"] = df["predicted_securities"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Explode the securities column so each security appears on its own row.
df_exploded = df.explode("predicted_securities")

# Count frequency for each security.
security_counts = df_exploded["predicted_securities"].value_counts()
top_n = 10
top_securities = security_counts.head(top_n).index.tolist()

# Filter for top securities.
df_top = df_exploded[df_exploded["predicted_securities"].isin(top_securities)]

# Group by security and sentiment, then pivot.
sec_sent_counts = df_top.groupby(["predicted_securities", "predicted_sentiment"]).size().reset_index(name="count")
security_pivot = sec_sent_counts.pivot(index="predicted_securities", columns="predicted_sentiment", values="count").fillna(0)

# Ensure all sentiment columns exist.
for col in ["positive", "negative", "neutral"]:
    if col not in security_pivot.columns:
        security_pivot[col] = 0

# Calculate total number of samples per security.
security_pivot["total"] = security_pivot.sum(axis=1)

# Calculate frequency densities (as percentages).
security_pivot["positive_density"] = security_pivot["positive"] / security_pivot["total"] * 100
security_pivot["negative_density"] = security_pivot["negative"] / security_pivot["total"] * 100
security_pivot["neutral_density"]  = security_pivot["neutral"]  / security_pivot["total"] * 100

# Compute net sentiment as percentage difference.
security_pivot["net_sentiment"] = (security_pivot["positive"] - security_pivot["negative"]) / security_pivot["total"] * 100

# Rank securities by net sentiment ascending.
security_pivot = security_pivot.sort_values("net_sentiment", ascending=True)

# Prepare data for plotting.
securities = security_pivot.index.tolist()
pos_sec_density = security_pivot["positive_density"]
neg_sec_density = security_pivot["negative_density"]
total_sec_samples = security_pivot["total"]
net_sec_sent = security_pivot["net_sentiment"]
neutral_sec_density = security_pivot["neutral_density"]

fig, ax2 = plt.subplots(figsize=(12, 8))
# Plot positive density for securities.
ax2.barh(securities, pos_sec_density, color="lightgreen", label="Positive")
# Plot negative density (as negative values).
ax2.barh(securities, -neg_sec_density, color="lightcoral", label="Negative")
# Draw vertical line at x=0.
ax2.axvline(0, color="black", linewidth=1)

# Annotate each bar conditionally.
for i, sec in enumerate(securities):
    ns = net_sec_sent.loc[sec]
    nd = neutral_sec_density.loc[sec]
    count = int(total_sec_samples.loc[sec])
    annotation = f"Net: {ns:.1f}% | Neutral: {nd:.1f}% (n={count})"
    if ns > 0:
        ax2.text(-offset, i, annotation, va='center', ha='left', color="Black", fontsize=12)
    elif ns < 0:
        ax2.text(offset, i, annotation, va='center', ha='right', color="Black", fontsize=12)
    else:
        ax2.text(0, i, annotation, va='center', ha='center', color="Black", fontsize=12)

ax2.set_title(f"Diverging Sentiment for Top {top_n} Predicted Themes (Frequency Density)", fontsize=18)
ax2.set_xlabel("Sentiment Frequency (%)", fontsize=16)
ax2.set_ylabel("Security", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.legend(fontsize=14, title="Sentiment", title_fontsize=16)
plt.tight_layout()
plt.show()
