```markdown
# IMDB Movie Review Sentiment Analysis Project

## Overview

This project explores sentiment analysis on the IMDB Large Movie Review Dataset[cite: 52]. It implements and compares a baseline machine learning model (Naive Bayes) with a fine-tuned Large Language Model (DistilBERT) [cite: 43] to classify movie reviews as positive or negative. The project includes scripts for exploratory data analysis (EDA), model training, evaluation, and an interactive Streamlit dashboard for demonstration[cite: 31, 37].

## Project Structure

The project follows this directory structure:

```
project/
├── code/
│   ├── run_eda.py
│   ├── test_baseline.py
│   ├── test_llm.py
│   ├── train_baseline.py
│   └── train_llm.py
├── data/
│   ├── aclImdb/
├── models/
│   ├── baselinemodel/
│   └── distilbert_fine_tuned/
├── results/
│   ├── baselinemodel/
│   ├── dataset_info/
│   ├── distilbert/
│   └── distilbert_training_output/
├── dashboard.py
├── README.md
└── requirements.txt
```

## Setup Instructions

**1. Clone the Repository (if applicable):**
   If this project is hosted on Git, clone it first:
   ```bash
   git clone https://github.com/KonetiBalaji/IMDB-Movie-Review-Sentiment-Analysis.git
   ```

**2. Create and Activate Virtual Environment:**
   It's highly recommended to use a virtual environment.
   ```bash
   # Create the environment (e.g., named '.venv')
   python -m venv .venv

   # Activate the environment
   # On Windows (Git Bash or cmd.exe)
   .venv\Scripts\activate
   # On macOS/Linux (bash/zsh)
   source .venv/bin/activate
   ```

**3. Install Dependencies:**
   Install all required Python libraries from `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

**4. Download NLTK Data:**
   The EDA script uses NLTK for tokenization and stopwords. Download the necessary data:
   ```bash
   python -m nltk.downloader punkt stopwords
   ```

**5. Download and Prepare IMDB Dataset:**
   * Download the "Large Movie Review Dataset v1.0" from [Stanford AI Labs](https://ai.stanford.edu/~amaas/data/sentiment/)[cite: 56]. The file is usually named `aclImdb_v1.tar.gz`.
   * Extract the downloaded archive.
   * Place the extracted `aclImdb` folder *inside* the `data/` directory in your project. The final path should look like: `project/data/aclImdb/`.

## Running the Project

All commands below should be run from the root directory (`project/`) with the virtual environment activated.

**1. Exploratory Data Analysis (EDA):**
   Generates insights about the dataset and saves summary/plots to `results/dataset_info/`.
   *(Assuming you saved the EDA script as `run_eda.py`)*
   ```bash
   python code/run_eda.py
   ```

**2. Train Baseline Model (Naive Bayes):**
   Trains the Naive Bayes model, saves the model/vectorizer to `models/baselinemodel/`, and saves results to `results/baselinemodel/`.
   *(Assuming you saved the script as `train_baseline.py`)*
   ```bash
   python code/train_baseline.py
   ```

**3. Fine-tune LLM (DistilBERT):**
   Fine-tunes the DistilBERT model. Saves the fine-tuned model to `models/distilbert_fine_tuned/` and results to `results/distilbert/`.
   *(Assuming you saved the script as `train_llm.py`)*
   **Note:** This step is computationally intensive and will take a very long time without a GPU. The script uses data sampling by default to speed this up. You can configure the sample size within the script.
   ```bash
   python code/train_llm.py
   ```

**4. Run the Interactive Dashboard:**
   Launches the Streamlit application in your browser, allowing you to test both models.
   *(Assuming you saved the script as `dashboard.py`)*
   ```bash
   streamlit run dashboard.py
   ```
   Open the URL provided in the terminal (usually `http://localhost:8501`).

## Output

* **Models:** Trained models are saved in the `models/` directory under respective subfolders (`baselinemodel`, `distilbert_fine_tuned`).
* **Results:** Evaluation metrics (`.txt`) and confusion matrix plots (`.png`) are saved in the `results/` directory under respective subfolders (`baselinemodel`, `distilbert`).
* **EDA:** Dataset insights are saved in `results/dataset_info/`.

```
