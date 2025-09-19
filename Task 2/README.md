# 📊 NLTK-Powered Text Analytics Web App

An interactive Streamlit web app for performing text preprocessing and NLP tasks such as tokenization, stopword removal, POS tagging, and lemmatization.  

## 🚀 Features
- Upload text input (or paste text directly)
- Preprocess with:
  - Cleaning (lowercasing, punctuation removal)
  - Tokenization
  - Stopword removal
  - Lemmatization
  - POS tagging
- Generate structured DataFrame outputs for each stage
- Interactive, web-based interface powered by Streamlit

## 🛠️ Tech Stack
- Python 3.10+
- Streamlit for web app UI
- NLTK for NLP preprocessing
- Pandas for tabular representation


## 📂 Project Structure
```
├── streamlit_app.py   # Main Streamlit application
├── nlp_pipeline.py    # NLP pipeline functions
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## ⚡ Installation & Usage

1. Clone the repository:
   ```bash
   git clone <your-repo-link>.git
   cd <repo-folder>
    ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
5. Open the local server URL (usually http://localhost:8501) in your browser.

## 📦 Requirements
Make sure the following are installed (also listed in requirements.txt):
- streamlit
- nltk
- pandas
Additionally, ensure NLTK resources are downloaded:
```
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')
```

## 📝 Example
Input text:
```
Natural Language Processing is amazing!
```
Output DataFrame includes:
- Tokens → ['natural', 'language', 'processing', 'amazing']
- POS Tags → [('natural', 'JJ'), ('language', 'NN'), ...]
- Lemmas → ['natural', 'language', 'process', 'amazing']

## 📌 Future Enhancements

- Named Entity Recognition (NER)
- Sentiment Analysis
- Word cloud visualization
- Support for multiple languages