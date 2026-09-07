# pip install gensim pyLDAvis wordcloud nltk

import re
import nltk
import gensim
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.models import LsiModel
from wordcloud import WordCloud

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Sample customer-support data
tickets = [
    "Customer charged twice for monthly bill",
    "Internet bill is incorrect and payment failed",
    "Network outage in my area since yesterday",
    "Mobile network is very slow and calls are dropping",
    "Phone device is overheating and battery drains quickly",
    "Customer device is not connecting to mobile network",
    "Unexpected charges appeared on my telecom bill",
    "No network signal after recent outage",
    "Mobile phone screen and battery have hardware issues"
]

# Text preprocessing
def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = text.split()
    return [w for w in tokens if w not in stop_words and len(w) > 2]

texts = [preprocess(ticket) for ticket in tickets]

# Dictionary and Bag-of-Words
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# -------- LSA / SVD --------
lsi = LsiModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3
)

print("LSA Topics:")
for topic in lsi.print_topics(num_words=5):
    print(topic)

# -------- LDA Topic Tuning --------
scores = []

for k in range(2, 6):
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=42,
        passes=10
    )

    coherence = CoherenceModel(
        model=lda,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v"
    ).get_coherence()

    scores.append((k, coherence))

# Select best number of topics
best_k = max(scores, key=lambda x: x[1])[0]
print("Best Topics:", best_k)

# Train final LDA model
lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=best_k,
    random_state=42,
    passes=20
)

# Display discovered topics
for i, topic in lda.print_topics(num_words=8):
    print(f"Topic {i}: {topic}")

# -------- Coherence Visualization --------
plt.plot([x[0] for x in scores], [x[1] for x in scores], marker="o")
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.title("LDA Topic Optimization")
plt.show()

# -------- WordCloud --------
for topic_id in range(best_k):
    words = dict(lda.show_topic(topic_id, topn=20))
    wc = WordCloud(width=800, height=400).generate_from_frequencies(words)
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Topic {topic_id}")
    plt.show()

# -------- pyLDAvis --------
vis = pyLDAvis.gensim_models.prepare(
    lda, corpus, dictionary
)
pyLDAvis.display(vis)

# -------- Ticket Topic Assignment --------
for i, ticket in enumerate(tickets):
    topic = max(lda[corpus[i]], key=lambda x: x[1])
    print(f"Ticket: {ticket}")
    print(f"Topic: {topic[0]}, Probability: {topic[1]:.2f}")