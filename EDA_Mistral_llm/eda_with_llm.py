# eda_with_llm.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline

# ------------------------------
# Step 1: Load Iris Dataset
# ------------------------------
df = pd.read_csv("c:\Users\Geetha\OneDrive\Documents\Datasets\Iris.csv")
sample_data = df.head(10).to_string()

# ------------------------------
# Step 2: Basic EDA
# ------------------------------
print("\n--- Basic Information ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Class Distribution ---")
print(df['species'].value_counts())

# ------------------------------
# Step 3: Data Visualization
# ------------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='species', y='sepal_length')
plt.title("Sepal Length by Species")
plt.savefig("sepal_length_by_species.png")
plt.show()

# ------------------------------
# Step 4: Load Mistral (GenAI) using Hugging Face
# ------------------------------
print("\n--- Loading Mistral Model ---")

try:
    llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

    prompt = f"""
    You are a data analyst. Perform EDA (exploratory data analysis) on this dataset:
    {sample_data}
    
    Please include:
    - Column types
    - Missing values
    - Distribution of species
    - Any interesting patterns
    - Suggest visualizations or trends
    """

    print("\n--- Asking Mistral to Generate EDA Summary ---")
    response = llm(prompt, max_new_tokens=300)[0]["generated_text"]
    print("\n--- Mistral's Response ---")
    print(response)

    # Save the output to a report
    with open("eda_report.txt", "w") as f:
        f.write(response)

except Exception as e:
    print(f"Could not load Mistral. Try on Google Colab.\nError: {e}")

# ------------------------------
# Step 5: Done
# ------------------------------
print("\nEDA project with Iris dataset complete.")

