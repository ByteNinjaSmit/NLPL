# pip install gensim pyLDAvis wordcloud nltk pandas matplotlib
#
# Topic Modeling on Telecom Customer-Support Tickets
# -------------------------------------------------
# Goal: automatically discover recurring complaint themes (billing, network
# outage, device issues) from support tickets and deliver an actionable
# topic-trend dashboard for the operations team.
#
# Pipeline:
#   1. Preprocess ticket text
#   2. Build dictionary + BoW / TF-IDF corpora
#   3. LSA / SVD topics (Gensim LsiModel)
#   4. LDA with topic-count tuning via c_v coherence
#   5. Auto-label topics from their top terms
#   6. Visualize: WordClouds, coherence curve, pyLDAvis
#   7. Assign every ticket a topic and export a topic-trend dashboard

import os
import re
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless: write image files, no GUI needed
import matplotlib.pyplot as plt
import pandas as pd

import nltk
import gensim
import pyLDAvis
import pyLDAvis.gensim_models

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel, LsiModel, TfidfModel, CoherenceModel
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

OUTPUT_DIR = "assignment4_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Sample dataset: telecom customer-support tickets
#    (date field lets us build a real complaint-trend view)
# ----------------------------------------------------------------------
raw_tickets = [
    ("2026-06-01", "Customer charged twice for the monthly bill this cycle"),
    ("2026-06-01", "Internet bill amount is incorrect and my online payment failed"),
    ("2026-06-02", "Unexpected roaming charges appeared on my telecom bill"),
    ("2026-06-02", "I was billed for a data pack I never activated or requested"),
    ("2026-06-03", "Autopay deducted the wrong amount from my bank account"),
    ("2026-06-03", "Refund for the duplicate payment has not been credited yet"),
    ("2026-06-04", "My plan price increased without any prior notice on the invoice"),
    ("2026-06-05", "Late payment fee added even though I paid the bill on time"),
    ("2026-06-06", "Prepaid recharge failed but the money was debited from my wallet"),
    ("2026-06-07", "Bill shows premium subscription charges I did not sign up for"),

    ("2026-06-02", "Complete network outage in my area since yesterday evening"),
    ("2026-06-03", "No network signal at home after the recent tower maintenance"),
    ("2026-06-04", "Mobile data is extremely slow and web pages will not load"),
    ("2026-06-04", "Calls keep dropping every few minutes on the mobile network"),
    ("2026-06-05", "Broadband connection down for three days and still not fixed"),
    ("2026-06-06", "Frequent network disconnections during peak evening hours"),
    ("2026-06-07", "Weak signal and poor call quality across the whole neighborhood"),
    ("2026-06-08", "Internet keeps disconnecting every hour on the fiber line"),
    ("2026-06-09", "Regional outage reported, no coverage in the entire district"),
    ("2026-06-10", "Slow upload speed and high latency on the home broadband"),
    ("2026-06-11", "Network completely unavailable after the power cut in our locality"),
    ("2026-06-12", "SMS and calls not working although data connection is fine"),

    ("2026-06-03", "Phone device is overheating and the battery drains very quickly"),
    ("2026-06-05", "New handset screen flickers and shows vertical lines"),
    ("2026-06-06", "SIM card is not detected by my mobile device anymore"),
    ("2026-06-07", "Router provided by the company keeps restarting on its own"),
    ("2026-06-08", "Phone will not connect to the mobile network after a software update"),
    ("2026-06-09", "Device speaker stopped working and audio is distorted on calls"),
    ("2026-06-10", "Set top box shows no signal error and will not boot up"),
    ("2026-06-11", "Battery health dropped fast and the phone shuts down at fifty percent"),
    ("2026-06-12", "Hardware fault in the modem, indicator light stays red constantly"),
    ("2026-06-13", "Touch screen unresponsive on the handset bought last month"),

    ("2026-06-08", "Charged twice again this month, billing system keeps double charging"),
    ("2026-06-09", "Wrong tax amount on the invoice inflates my total bill"),
    ("2026-06-10", "Data balance deducted even when connected to home wifi"),
    ("2026-06-11", "No internet and no signal since the storm damaged the tower"),
    ("2026-06-12", "Mobile network unstable, speed test shows almost zero bandwidth"),
    ("2026-06-13", "Phone battery swelling and the back cover is coming off"),
    ("2026-06-14", "Company modem overheating and disconnecting several times a day"),
    ("2026-06-14", "Payment portal error prevents me from clearing my outstanding bill"),
    ("2026-06-15", "Account suspended for non payment but my bill was already paid"),
    ("2026-06-15", "Coverage dropped to a single bar across the city after upgrade"),
    ("2026-06-16", "New phone microphone not working, callers cannot hear my voice"),
]

tickets_df = pd.DataFrame(raw_tickets, columns=["date", "text"])
tickets_df["date"] = pd.to_datetime(tickets_df["date"])
print(f"Loaded {len(tickets_df)} support tickets "
      f"({tickets_df['date'].min().date()} to {tickets_df['date'].max().date()})")

# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()

base_stopwords = set(stopwords.words("english"))
# domain words that are frequent but carry no topic signal
domain_stopwords = {
    "customer", "please", "would", "could", "also", "get", "got", "since",
    "still", "even", "though", "although", "month", "monthly", "cycle",
    "area", "home", "day", "days", "time", "keep", "keeps", "company",
    "provided", "new", "old", "last", "several", "across", "entire", "whole",
}
STOP_WORDS = base_stopwords | domain_stopwords


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


texts = [preprocess(t) for t in tickets_df["text"]]

# ----------------------------------------------------------------------
# 3. Dictionary + corpora (BoW and TF-IDF)
# ----------------------------------------------------------------------
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=2, no_above=0.5)

bow_corpus = [dictionary.doc2bow(t) for t in texts]

tfidf_model = TfidfModel(bow_corpus)
tfidf_corpus = tfidf_model[bow_corpus]

print(f"Vocabulary size after filtering: {len(dictionary)}")

# ----------------------------------------------------------------------
# 4. LSA / SVD  (LSA works better on TF-IDF weighted input)
# ----------------------------------------------------------------------
LSA_TOPICS = 3
lsa_model = LsiModel(
    corpus=tfidf_corpus,
    id2word=dictionary,
    num_topics=LSA_TOPICS,
)

print("\n" + "=" * 60)
print("LSA / SVD TOPICS")
print("=" * 60)
for topic_id, topic in lsa_model.print_topics(num_words=6):
    print(f"Topic {topic_id}: {topic}")

# ----------------------------------------------------------------------
# 5. LDA topic-count tuning via c_v coherence
# ----------------------------------------------------------------------
def build_lda(num_topics, passes=15):
    return LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=RANDOM_STATE,
        passes=passes,
        iterations=200,
        alpha="auto",
        eta="auto",
    )


def coherence_cv(model):
    return CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    ).get_coherence()


K_RANGE = range(2, 8)
coherence_scores = []

print("\n" + "=" * 60)
print("LDA COHERENCE TUNING (c_v)")
print("=" * 60)
for k in K_RANGE:
    model_k = build_lda(k)
    score = coherence_cv(model_k)
    coherence_scores.append((k, score))
    print(f"num_topics = {k:2d}  ->  coherence = {score:.4f}")

best_k = max(coherence_scores, key=lambda x: x[1])[0]
print(f"\nBest number of topics: {best_k}")

# ----------------------------------------------------------------------
# 6. Final LDA model + auto topic labels
# ----------------------------------------------------------------------
lda_model = build_lda(best_k, passes=30)

# heuristic keyword -> theme mapping for readable dashboard labels
THEME_KEYWORDS = {
    "Billing & Payments": {
        "bill", "charge", "charged", "payment", "invoice", "refund", "recharge",
        "paid", "fee", "amount", "tax", "autopay", "wallet", "account", "pack",
    },
    "Network Outage & Coverage": {
        "network", "outage", "signal", "coverage", "tower", "slow", "speed",
        "disconnect", "disconnecting", "connection", "broadband", "fiber",
        "latency", "bandwidth", "call", "drop", "dropping", "internet", "data",
    },
    "Device & Hardware": {
        "phone", "device", "battery", "screen", "handset", "router", "modem",
        "sim", "hardware", "overheating", "speaker", "microphone", "touch",
        "box", "boot", "update", "charging",
    },
}


def label_topic(top_terms):
    term_set = {t for t, _ in top_terms}
    best_theme, best_hits = "Other / Mixed", 0
    for theme, keywords in THEME_KEYWORDS.items():
        hits = len(term_set & keywords)
        if hits > best_hits:
            best_theme, best_hits = theme, hits
    return best_theme


topic_labels = {}
print("\n" + "=" * 60)
print("DISCOVERED LDA TOPICS")
print("=" * 60)
for topic_id in range(best_k):
    top_terms = lda_model.show_topic(topic_id, topn=10)
    label = label_topic(top_terms)
    # de-duplicate labels if two topics map to the same theme
    if label in topic_labels.values():
        label = f"{label} ({topic_id})"
    topic_labels[topic_id] = label
    terms_str = ", ".join(f"{t}" for t, _ in top_terms)
    print(f"Topic {topic_id}  [{label}]")
    print(f"   {terms_str}\n")

# ----------------------------------------------------------------------
# 7a. Visualization: coherence curve
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot([k for k, _ in coherence_scores],
         [s for _, s in coherence_scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
plt.xlabel("Number of Topics")
plt.ylabel("c_v Coherence Score")
plt.title("LDA Topic-Count Optimization")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
coherence_path = os.path.join(OUTPUT_DIR, "coherence_curve.png")
plt.savefig(coherence_path, dpi=120)
plt.close()
print(f"Saved {coherence_path}")

# ----------------------------------------------------------------------
# 7b. Visualization: per-topic WordClouds
# ----------------------------------------------------------------------
n_cols = min(best_k, 3)
n_rows = (best_k + n_cols - 1) // n_cols
plt.figure(figsize=(6 * n_cols, 4 * n_rows))
for topic_id in range(best_k):
    freqs = dict(lda_model.show_topic(topic_id, topn=25))
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="viridis").generate_from_frequencies(freqs)
    plt.subplot(n_rows, n_cols, topic_id + 1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Topic {topic_id}: {topic_labels[topic_id]}")
plt.tight_layout()
wordcloud_path = os.path.join(OUTPUT_DIR, "topic_wordclouds.png")
plt.savefig(wordcloud_path, dpi=120)
plt.close()
print(f"Saved {wordcloud_path}")

# ----------------------------------------------------------------------
# 7c. Visualization: pyLDAvis interactive HTML
# ----------------------------------------------------------------------
vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary)
ldavis_path = os.path.join(OUTPUT_DIR, "pyldavis.html")
pyLDAvis.save_html(vis, ldavis_path)
print(f"Saved {ldavis_path}")

# ----------------------------------------------------------------------
# 8. Assign a dominant topic to every ticket
# ----------------------------------------------------------------------
def dominant_topic(bow):
    dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
    topic_id, prob = max(dist, key=lambda x: x[1])
    return topic_id, prob


assignments = [dominant_topic(bow) for bow in bow_corpus]
tickets_df["topic_id"] = [a[0] for a in assignments]
tickets_df["topic_prob"] = [round(a[1], 3) for a in assignments]
tickets_df["topic_label"] = tickets_df["topic_id"].map(topic_labels)

print("\n" + "=" * 60)
print("SAMPLE TICKET -> TOPIC ASSIGNMENTS")
print("=" * 60)
print(tickets_df[["date", "text", "topic_label", "topic_prob"]]
      .head(12).to_string(index=False))

assignments_path = os.path.join(OUTPUT_DIR, "ticket_topic_assignments.csv")
tickets_df.to_csv(assignments_path, index=False)
print(f"\nSaved {assignments_path}")

# ----------------------------------------------------------------------
# 9. Operations dashboard: complaint volume + weekly trend
# ----------------------------------------------------------------------
volume = (tickets_df["topic_label"].value_counts()
          .rename_axis("topic").reset_index(name="tickets"))
volume["share_%"] = (100 * volume["tickets"] / len(tickets_df)).round(1)

tickets_df["week"] = tickets_df["date"].dt.to_period("W").dt.start_time
trend = (tickets_df.groupby(["week", "topic_label"]).size()
         .unstack(fill_value=0).sort_index())

trend_path = os.path.join(OUTPUT_DIR, "topic_trends.csv")
trend.to_csv(trend_path)
volume_path = os.path.join(OUTPUT_DIR, "topic_volume.csv")
volume.to_csv(volume_path, index=False)

print("\n" + "=" * 60)
print("COMPLAINT THEME VOLUME")
print("=" * 60)
print(volume.to_string(index=False))
print("\nWEEKLY TOPIC TREND")
print(trend.to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(volume["topic"], volume["tickets"], color="steelblue")
axes[0].set_title("Complaint Volume by Theme")
axes[0].set_ylabel("Number of Tickets")
axes[0].tick_params(axis="x", rotation=20)
for i, v in enumerate(volume["tickets"]):
    axes[0].text(i, v + 0.1, str(v), ha="center")

for label in trend.columns:
    axes[1].plot(trend.index, trend[label], marker="o", label=label)
axes[1].set_title("Weekly Complaint Trend by Theme")
axes[1].set_ylabel("Tickets per Week")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Telecom Support Tickets - Topic Trend Dashboard", fontsize=14)
fig.tight_layout()
dashboard_path = os.path.join(OUTPUT_DIR, "topic_dashboard.png")
fig.savefig(dashboard_path, dpi=120)
plt.close(fig)
print(f"\nSaved {dashboard_path}")

# ----------------------------------------------------------------------
# 10. Actionable summary for the operations team
# ----------------------------------------------------------------------
top_theme = volume.iloc[0]
latest_week = trend.index.max()
prev_week = trend.index[-2] if len(trend.index) > 1 else latest_week
delta = (trend.loc[latest_week] - trend.loc[prev_week]).sort_values(ascending=False)
rising = delta.index[0]

print("\n" + "=" * 60)
print("OPERATIONS SUMMARY")
print("=" * 60)
print(f"- Dominant complaint theme: {top_theme['topic']} "
      f"({top_theme['tickets']} tickets, {top_theme['share_%']}% of volume)")
print(f"- Fastest rising theme in the latest week: {rising} "
      f"(+{int(delta.iloc[0])} vs previous week)")
print(f"- Deliverables in ./{OUTPUT_DIR}/ : "
      f"topic_dashboard.png, topic_trends.csv, topic_volume.csv, "
      f"ticket_topic_assignments.csv, topic_wordclouds.png, "
      f"coherence_curve.png, pyldavis.html")
