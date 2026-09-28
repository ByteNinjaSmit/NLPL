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
import numpy as np
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
# Dates span 5 weeks. The data has a deliberate trend for the ops story:
# billing complaints start high then fall, device stays flat, network
# outage complaints climb week over week (a spreading infrastructure issue).
raw_tickets = [
    # ---- Billing & Payments  (front-loaded, declining) ----
    ("2026-06-01", "Billing error charged me twice for the same monthly bill"),
    ("2026-06-01", "My telecom bill amount is incorrect and far higher than my plan"),
    ("2026-06-02", "Unexpected roaming charges added to my bill this billing cycle"),
    ("2026-06-02", "I was billed for a data pack I never activated on my account"),
    ("2026-06-03", "Online bill payment failed but the amount was debited from my account"),
    ("2026-06-04", "Refund for the duplicate bill payment has not been credited"),
    ("2026-06-05", "Late payment fee charged on my bill even though I paid on time"),
    ("2026-06-09", "Prepaid recharge payment failed and the money was not refunded"),
    ("2026-06-10", "Bill shows a premium subscription charge I never signed up for"),
    ("2026-06-11", "Wrong tax amount on the invoice inflated my total bill"),
    ("2026-06-17", "Autopay charged the wrong bill amount from my bank account"),
    ("2026-06-24", "Account suspended for non payment but my bill was already paid"),

    # ---- Network Outage & Coverage  (climbing week over week) ----
    ("2026-06-03", "Complete network outage in my area since yesterday evening"),
    ("2026-06-06", "No network signal at home after the tower maintenance"),
    ("2026-06-09", "Mobile network data is extremely slow and pages will not load"),
    ("2026-06-10", "Calls keep dropping on the mobile network every few minutes"),
    ("2026-06-15", "Network outage across the whole locality, no signal on any phone"),
    ("2026-06-16", "Frequent network disconnections during peak evening hours"),
    ("2026-06-17", "Weak network signal and poor call quality in my neighborhood"),
    ("2026-06-19", "Regional network outage, no coverage in the entire district"),
    ("2026-06-22", "Network completely down after the power cut in our locality"),
    ("2026-06-23", "Calls and messages not working, network signal keeps disappearing"),
    ("2026-06-24", "No network and no signal since the storm damaged the tower"),
    ("2026-06-25", "Network coverage dropped to a single bar across the city"),
    ("2026-06-26", "Long network outage overnight, no signal until the morning"),
    ("2026-06-29", "Repeated network outage in the sector, tower still not restored"),
    ("2026-06-30", "No network signal for two days after the regional outage"),
    ("2026-07-01", "Total network blackout in the area, no coverage on any device"),

    # ---- Device & Hardware  (roughly flat) ----
    ("2026-06-02", "Phone battery drains very quickly and the device keeps overheating"),
    ("2026-06-07", "New handset screen flickers and shows vertical lines"),
    ("2026-06-08", "SIM card is not detected by my phone device anymore"),
    ("2026-06-12", "Router hardware keeps restarting on its own every hour"),
    ("2026-06-14", "Phone screen is unresponsive to touch after a software update"),
    ("2026-06-18", "Device speaker stopped working and call audio is distorted"),
    ("2026-06-20", "Set top box hardware shows an error and will not boot up"),
    ("2026-06-21", "Phone battery health dropped fast and the device shuts down early"),
    ("2026-06-27", "Modem hardware fault, the indicator light stays red constantly"),
    ("2026-06-28", "Phone battery is swelling and the back cover is coming off"),
    ("2026-07-02", "Handset touch screen is dead on the phone bought last month"),
    ("2026-07-03", "Charger port on the phone device is loose and will not charge"),
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
def build_lda(num_topics, passes=15, random_state=RANDOM_STATE):
    return LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        iterations=400,
        # short tickets -> few topics each -> sparse doc-topic prior
        alpha=0.1,
        eta="auto",
    )


def coherence_cv(model):
    return CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,  # avoid multiprocessing (Windows spawn needs __main__ guard)
    ).get_coherence()


def best_lda(num_topics, restarts=8, passes=40):
    """LDA on a tiny corpus is seed-sensitive; keep the most coherent restart."""
    best_model, best_score = None, -1.0
    for seed in range(restarts):
        model = build_lda(num_topics, passes=passes, random_state=seed)
        score = coherence_cv(model)
        if score > best_score:
            best_model, best_score = model, score
    return best_model, best_score


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

peak_k, peak_score = max(coherence_scores, key=lambda x: x[1])
# Business constraint: operations expects a small, interpretable set of
# complaint themes. Search the full range for the coherence curve, but pick
# the final k from the operationally useful window [MIN_K, MAX_K] - the
# highest-coherence k that a human dashboard can still read.
MIN_K, MAX_K = 3, 5
candidates = [(k, s) for k, s in coherence_scores if MIN_K <= k <= MAX_K]
best_k = max(candidates, key=lambda x: x[1])[0]
print(f"\nPeak coherence at k = {peak_k} ({peak_score:.4f})")
print(f"Selected k = {best_k} "
      f"(best coherence within the readable range {MIN_K}-{MAX_K})")

# ----------------------------------------------------------------------
# 6. Final LDA model + auto topic labels
# ----------------------------------------------------------------------
lda_model, final_coherence = best_lda(best_k)
print(f"Final LDA: k = {best_k}, c_v coherence = {final_coherence:.4f}")

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
    # score each theme by the total probability mass its keywords hold in
    # the topic (weight-aware, so a rare keyword at rank 10 counts little)
    scores = {theme: 0.0 for theme in THEME_KEYWORDS}
    for term, weight in top_terms:
        for theme, keywords in THEME_KEYWORDS.items():
            if term in keywords:
                scores[theme] += weight
    best_theme, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_theme if best_score > 0 else "Other / Mixed"


# Several fine-grained LDA topics may map to the same business theme.
# That is intentional: the dashboard groups tickets by theme, not raw topic id.


topic_labels = {}
print("\n" + "=" * 60)
print("DISCOVERED LDA TOPICS")
print("=" * 60)
for topic_id in range(best_k):
    top_terms = lda_model.show_topic(topic_id, topn=10)
    label = label_topic(top_terms)
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
plt.axvline(peak_k, color="orange", linestyle=":", label=f"peak k = {peak_k}")
plt.axvline(best_k, color="red", linestyle="--", label=f"selected k = {best_k}")
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
ldavis_path = os.path.join(OUTPUT_DIR, "pyldavis.html")
try:
    # mmds avoids the PCoA path that can emit complex values on new numpy/scipy
    vis = pyLDAvis.gensim_models.prepare(
        lda_model, bow_corpus, dictionary, mds="mmds"
    )
    pyLDAvis.save_html(vis, ldavis_path)
    print(f"Saved {ldavis_path}")
except Exception as exc:  # keep the dashboard pipeline running regardless
    print(f"pyLDAvis skipped ({type(exc).__name__}: {exc})")

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

# The first and last calendar weeks of a ticket export are almost always
# partial. Drop them when there is enough history so the trend is not skewed.
if len(trend) >= 4:
    full_weeks = trend.iloc[1:-1]
else:
    full_weeks = trend

print("\n" + "=" * 60)
print("OPERATIONS SUMMARY")
print("=" * 60)
print(f"- Dominant complaint theme: {top_theme['topic']} "
      f"({top_theme['tickets']} tickets, {top_theme['share_%']}% of volume)")

if len(full_weeks) >= 2:
    # average change per week from a linear fit over the full weeks
    x = list(range(len(full_weeks)))
    slopes = {theme: float(np.polyfit(x, full_weeks[theme].values, 1)[0])
              for theme in full_weeks.columns}
    theme, slope = max(slopes.items(), key=lambda kv: kv[1])
    window = f"{full_weeks.index[0].date()} -> {full_weeks.index[-1].date()}"
    if slope > 0.3:
        print(f"- Rising theme ({window}): {theme} (+{slope:.1f} tickets/week) "
              f"-> escalate to the network operations team")
    else:
        print(f"- No theme is clearly trending up over {window}; "
              f"largest mover is {theme} ({slope:+.1f} tickets/week)")
else:
    print("- Trend: not enough full weeks of history to compute a direction")
print(f"- Deliverables in ./{OUTPUT_DIR}/ : "
      f"topic_dashboard.png, topic_trends.csv, topic_volume.csv, "
      f"ticket_topic_assignments.csv, topic_wordclouds.png, "
      f"coherence_curve.png, pyldavis.html")
