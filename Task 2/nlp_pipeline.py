# nlp_pipeline.py
import re
from typing import List, Tuple, Dict
from collections import Counter
import pandas as pd
import nltk

# Ensure required resources are available
resources = [
    'punkt', 
    'punkt_tab', 
    'averaged_perceptron_tagger', 
    'averaged_perceptron_tagger_eng',  # NEW
    'wordnet', 
    'stopwords'
]

for r in resources:
    try:
        nltk.data.find(r)
    except LookupError:
        nltk.download(r)


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.sentiment import SentimentIntensityAnalyzer

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
SIA = SentimentIntensityAnalyzer()
BIGRAM_MEASURES = BigramAssocMeasures()

_WORD_RE = re.compile(r"[A-Za-z']{2,}")

def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace('\n', ' ')
    # remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # keep only letters and apostrophes (basic)
    return text

def tokenize(text: str) -> List[str]:
    toks = word_tokenize(text)
    toks = [t.lower() for t in toks if _WORD_RE.match(t)]
    return toks

def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]

def lemmatize(tokens: List[str]) -> List[str]:
    # simple mapping: lemma per token
    return [LEMMATIZER.lemmatize(t) for t in tokens]

def pos_tags(tokens: List[str]) -> List[Tuple[str, str]]:
    return pos_tag(tokens)

def sentiment_scores(text: str) -> Dict[str, float]:
    return SIA.polarity_scores(text)

def freq_dist(tokens: List[str], n: int = 30) -> List[Tuple[str,int]]:
    c = Counter(tokens)
    return c.most_common(n)

def top_ngrams(tokens: List[str], ngram=2, top_n=20) -> List[Tuple[Tuple[str,...], int]]:
    if ngram == 1:
        return freq_dist(tokens, top_n)
    # for bigrams (ngram==2) use collocation finder for more interesting pairs
    if ngram == 2:
        finder = BigramCollocationFinder.from_words(tokens)
        # score by raw frequency
        scored = finder.ngram_fd.items()  # dict-like
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    # fallback: naive ngram
    ngrams = zip(*(tokens[i:] for i in range(ngram)))
    c = Counter(ngrams)
    return c.most_common(top_n)

def collocations(tokens: List[str], top_n=20) -> List[Tuple[Tuple[str,str], float]]:
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(BIGRAM_MEASURES.pmi)
    return scored[:top_n]

def build_dataframe(texts: List[str]) -> pd.DataFrame:
    """
    texts: list of raw document strings
    Returns a pandas DataFrame with columns:
    - doc_id, text, clean_text, tokens, tokens_no_stop, lemmas, pos_tags, sentiment
    """
    rows = []
    for i, raw in enumerate(texts):
        cleaned = clean_text(raw)
        toks = tokenize(cleaned)
        toks_ns = remove_stopwords(toks)
        lemmas = lemmatize(toks_ns)
        pos = pos_tags(toks_ns)
        sentiment = sentiment_scores(cleaned)
        rows.append({
            'doc_id': i,
            'text': raw,
            'clean_text': cleaned,
            'tokens': toks,
            'tokens_no_stop': toks_ns,
            'lemmas': lemmas,
            'pos_tags': pos,
            'sentiment': sentiment
        })
    df = pd.DataFrame(rows)
    return df

def corpus_level_stats(df: pd.DataFrame) -> Dict:
    """
    Compute aggregated stats over the DataFrame
    Returns dict with:
      - total_docs, total_tokens, vocab_size, top_unigrams, top_bigrams
      - avg_compound_sentiment, sentiment_by_doc (list)
    """
    all_tokens = []
    all_lemmas = []
    sentiment_compounds = []
    for _, row in df.iterrows():
        all_tokens.extend(row['tokens_no_stop'])
        all_lemmas.extend(row['lemmas'])
        sentiment_compounds.append(row['sentiment']['compound'])

    total_docs = len(df)
    total_tokens = len(all_tokens)
    vocab_size = len(set(all_lemmas))
    top_unigrams = freq_dist(all_lemmas, n=30)
    top_bigrams = top_ngrams(all_tokens, ngram=2, top_n=30)
    avg_compound = sum(sentiment_compounds) / total_docs if total_docs > 0 else 0.0

    return {
        'total_docs': total_docs,
        'total_tokens': total_tokens,
        'vocab_size': vocab_size,
        'top_unigrams': top_unigrams,
        'top_bigrams': top_bigrams,
        'avg_compound_sentiment': avg_compound,
        'sentiment_series': sentiment_compounds
    }
