
📊 EDA Automation with Mistral using LLaMA — Iris Dataset
🔍 Project Objective
This project demonstrates how to automate Exploratory Data Analysis (EDA) using a large language model (LLaMA, Mistral variant) on the Iris flower dataset. It produces textual summaries and insights without manual interpretation, using HuggingFace's Transformers.

📁 Dataset
Dataset: Iris Dataset (CSV format)

Source: UCI Machine Learning Repository

Features:

SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm

Species (Target class)

🚀 Technologies Used
Python

Jupyter Notebook

Pandas, Matplotlib, Seaborn

Transformers (Mistral via Hugging Face)

LLaMA / Mistral LLM

Anaconda (for environment management)

📌 Project Structure
bash
Copy
Edit
EDA_Mistral_Iris/
│
├── iris.csv                  # Dataset
├── EDA_Automation_LLaMA.ipynb  # Main Jupyter notebook
├── eda_summary.txt           # LLM-generated natural language EDA
├── requirements.txt          # Libraries used in the project
└── README.md                 # Project overview (this file)
✅ Features of This Project
Loads the Iris dataset

Generates summary statistics and distribution plots

Automates EDA interpretation using a LLaMA-based model (Mistral)

Saves the summary into a .txt file

Useful for Data Science interviews and portfolio

💻 How to Run
Clone/download the project.

Open EDA_Automation_LLaMA.ipynb in Jupyter Notebook.

Run all cells step-by-step.

View the auto-generated eda_summary.txt.

📌 Sample Output
The Iris dataset contains 150 samples with 3 distinct species. Petal length and width are key in distinguishing the species. Setosa has smallest features, while Virginica has the largest.