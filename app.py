"""
app.py - Fake News Detector (Streamlit)
Run with:  py -m streamlit run app.py
"""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE, "src"))
from explain import analyze, fetch_article, highlight_html, verdict  # noqa: E402
from preprocess import clean_text  # noqa: E402

MODELS_DIR = os.path.join(BASE, "models")
OUTPUTS_DIR = os.path.join(BASE, "outputs")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return model, vectorizer, meta


try:
    model, vectorizer, meta = load_artifacts()
except FileNotFoundError:
    st.error("No trained model found. Run `py src/train.py` first.")
    st.stop()

EXAMPLES = {
    "Sensational example": (
        "SHOCKING!!! You won't believe what doctors don't want you to know! This one weird "
        "trick cures cancer overnight and the corrupt mainstream media is hiding it. WAKE UP!"
    ),
    "Factual example": (
        "The city council voted 6-3 on Tuesday to approve the new transit budget, according to "
        "a statement from the mayor's office. Officials said two bus routes will be extended "
        "next quarter. \"This investment reflects what residents asked for,\" council member "
        "Dana Lopez said. The study published in March by the transport authority found that "
        "ridership rose 12 percent over the past year, and data from the survey confirmed demand."
    ),
}


def render_analysis(text):
    res = analyze(text, model, vectorizer)
    label_text, kind = verdict(res["index"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Credibility Index", f"{res['index']}/100")
    c2.metric("ML prediction", res["label"], f"{res['p_real']*100:.0f}% REAL signal", delta_color="off")
    c3.metric("Sensationalism", f"{res['style']['sensationalism']}/100")
    getattr(st, kind)(f"**{label_text}**")
    if res["known_terms"] < 8:
        st.warning(f"Only {res['known_terms']} words were recognised by the model, so the ML result is "
                   "unreliable. Paste a longer passage (3+ sentences) for a better answer.")
    st.progress(res["index"] / 100)
    st.caption("Index = 70% ML model signal + 30% writing-style signal (heuristic weights, "
               "not a fact-check).")

    st.subheader("🧠 Why did the model decide this?")
    if res["contributions"]:
        st.markdown("<small>🟥 pushes toward **FAKE** &nbsp; 🟩 pushes toward **REAL** "
                    "(darker = stronger)</small>", unsafe_allow_html=True)
        st.markdown(highlight_html(text, res["contributions"]), unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown("**Top FAKE-leaning terms**")
        a.write(", ".join(t for t, _ in res["top_fake"]) or "-")
        b.markdown("**Top REAL-leaning terms**")
        b.write(", ".join(t for t, _ in res["top_real"]) or "-")
    else:
        st.info("This model type has no word weights to explain.")

    st.subheader("🚩 Writing-style red flags")
    if res["style"]["flags"]:
        for flag in res["style"]["flags"]:
            st.markdown(f"- {flag}")
    else:
        st.write("No major style red flags detected.")
    if res["style"]["sourcing"] is not None:
        st.write(f"Sourcing strength: **{res['style']['sourcing']}/100**")


st.title("📰 Fake News Detector")
st.caption(f"Model: **{meta['best_model_name']}** | test accuracy {meta['accuracy']:.1%} | "
           f"F1 {meta['f1_score']:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze text", "🌐 Analyze URL", "📂 Batch CSV", "📊 Model performance"])

with tab1:
    choice = st.selectbox("Load an example (optional)", ["-- none --"] + list(EXAMPLES))
    text = st.text_area("Paste a headline or full article", value=EXAMPLES.get(choice, ""), height=180)
    if st.button("Analyze", type="primary", key="go_text"):
        if text.strip():
            render_analysis(text)
        else:
            st.warning("Please enter some text.")

with tab2:
    url = st.text_input("Article URL", placeholder="https://...")
    if st.button("Fetch & analyze", type="primary", key="go_url"):
        try:
            with st.spinner("Fetching article..."):
                title, body = fetch_article(url)
            if len(body.split()) < 30:
                st.warning("Couldn't extract enough article text from that page (it may be paywalled "
                           "or JavaScript-rendered). Paste the text in the first tab instead.")
            else:
                st.write(f"**Fetched:** {title}")
                render_analysis(f"{title}\n{body}")
        except Exception as e:  # network / invalid URL
            st.error(f"Could not fetch that URL: {e}")

with tab3:
    st.write("Upload a CSV of articles to classify them all at once.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        default = list(df.columns).index("text") if "text" in df.columns else 0
        col = st.selectbox("Column containing the article text", df.columns, index=default)
        if st.button("Classify all rows", type="primary", key="go_batch"):
            texts = df[col].fillna("").astype(str)
            X = vectorizer.transform(texts.apply(clean_text))
            out = df.copy()
            out["prediction"] = model.predict(X)
            st.write(out["prediction"].value_counts())
            st.dataframe(out.head(200))
            st.download_button("Download results CSV", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

with tab4:
    comp = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if os.path.exists(comp):
        st.dataframe(pd.read_csv(comp).round(4), hide_index=True)
    img = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    if os.path.exists(img):
        st.image(img)

st.markdown("---")
st.caption("⚠️ Educational ML demo. It detects writing patterns, not factual truth - "
           "always verify news with trusted independent sources.")
