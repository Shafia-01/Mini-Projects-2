import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List
from nlp_pipeline import build_dataframe, corpus_level_stats, top_ngrams, collocations

st.set_page_config(page_title="NLTK Text Analytics", layout="wide")

st.sidebar.title("NLTK Text Analytics")
st.sidebar.markdown("Upload a text file (one document per line) or paste text.")
upload = st.sidebar.file_uploader("Upload .txt or .csv (one doc per line or a 'text' column)", type=['txt','csv'])
raw_text_area = st.sidebar.text_area("Or paste text (separate docs with blank line):", height=120)
min_tokens = st.sidebar.slider("Min tokens to include doc", 0, 5, 0)

@st.cache_data
def load_texts_from_upload(upload, raw_paste) -> List[str]:
    texts = []
    if upload is not None:
        try:
            if upload.type == "text/csv" or str(upload.name).lower().endswith('.csv'):
                df = pd.read_csv(upload)
                if 'text' in df.columns:
                    texts = df['text'].astype(str).tolist()
                else:
                    texts = df.iloc[:,0].astype(str).tolist()
            else:
                content = upload.getvalue().decode('utf-8')
                texts = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            st.sidebar.error(f"Could not read uploaded file: {e}")
    if raw_paste:
        parts = [p.strip() for p in raw_paste.split('\n\n') if p.strip()]
        if parts:
            texts = parts + texts
    if not texts:
        texts = [
            "I loved the movie. The acting was great, and the story moved me.",
            "Terrible service at the restaurant. Food was cold and staff were rude.",
            "Neutral review: the product works as expected but nothing special.",
        ]
    if min_tokens > 0:
        filtered = []
        for t in texts:
            if len(t.split()) >= min_tokens:
                filtered.append(t)
        texts = filtered
    return texts

texts = load_texts_from_upload(upload, raw_text_area)

with st.spinner("Running NLP pipeline..."):
    df = build_dataframe(texts)
    stats = corpus_level_stats(df)
    
tabs = st.tabs(["Data Explorer", "Analysis Dashboard"])

with tabs[0]:
    st.header("Data Explorer")
    st.markdown("View documents and basic NLP outputs.")
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader("Documents")
        for idx, row in df.iterrows():
            with st.expander(f"Doc {row['doc_id']} — preview"):
                st.write(row['text'])
                st.write("**Clean text:**", row['clean_text'])
                st.write("**Sentiment:**", row['sentiment'])
    with col2:
        st.subheader("Corpus Summary")
        st.metric("Documents", stats['total_docs'])
        st.metric("Total tokens (no-stop)", stats['total_tokens'])
        st.metric("Vocabulary size", stats['vocab_size'])
        st.markdown("**Avg compound sentiment**: {:.3f}".format(stats['avg_compound_sentiment']))

    st.subheader("Dataframe (preview)")
    st.dataframe(df[['doc_id','text','clean_text']].head(200))

with tabs[1]:
    st.header("Analysis Dashboard")
    st.markdown("Frequency distributions, collocations, n-grams, and sentiment trends.")

    st.subheader("Top Unigrams (lemmas)")
    top_uni = stats['top_unigrams'][:30]
    if top_uni:
        uni_df = pd.DataFrame(top_uni, columns=['word','count'])
        fig_uni = px.bar(uni_df, x='word', y='count', title='Top Unigrams (lemmas)')
        st.plotly_chart(fig_uni, use_container_width=True)
    else:
        st.info("No unigrams found.")

    st.subheader("Top Bigrams")
    top_bi = stats['top_bigrams'][:30]
    if top_bi:
        rows = []
        for item in top_bi:
            if isinstance(item[0], tuple):
                w1, w2 = item[0]
                cnt = item[1]
            else:
                try:
                    (w1, w2), cnt = item
                except:
                    continue
            rows.append({'bigram': f"{w1} {w2}", 'count': cnt})
        bi_df = pd.DataFrame(rows).sort_values('count', ascending=False).head(30)
        fig_bi = px.bar(bi_df, x='bigram', y='count', title='Top Bigrams')
        st.plotly_chart(fig_bi, use_container_width=True)

    st.subheader("Top Collocations (PMI)")
    colloc_list = collocations(sum(df['tokens_no_stop'], []), top_n=20)
    if colloc_list:
        colloc_df = pd.DataFrame([{"bigram": f"{a} {b}", "score": s} for (a,b), s in colloc_list])
        fig_coll = px.bar(colloc_df, x='bigram', y='score', title='Top Collocations (PMI)')
        st.plotly_chart(fig_coll, use_container_width=True)

    st.subheader("Sentiment trend (compound scores)")
    sent_series = stats['sentiment_series']
    sent_df = pd.DataFrame({'doc_id': list(range(len(sent_series))), 'compound': sent_series})
    fig_sent = px.line(sent_df, x='doc_id', y='compound', title='Compound Sentiment by Document', markers=True)
    st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader("Download processed data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV of processed DataFrame", data=csv, file_name='processed_nlp.csv', mime='text/csv')

st.sidebar.markdown("---")
st.sidebar.markdown("Built with NLTK + pandas + Streamlit")
