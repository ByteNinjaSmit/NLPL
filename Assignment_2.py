import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "machine learning is a branch of artificial intelligence",
    "deep learning is a part of machine learning",
    "artificial intelligence uses machine learning techniques",
    "data science uses machine learning and statistics"
]

vocabulary = sorted(set(" ".join(documents).split()))
word_index = {word: i for i, word in enumerate(vocabulary)}

bow = np.array([
    [doc.split().count(word) for word in vocabulary]
    for doc in documents
])

n_docs = len(documents)
tf = bow / bow.sum(axis=1, keepdims=True)
df = np.count_nonzero(bow, axis=0)
idf = np.log(n_docs / df)
tfidf = tf * idf
tfidf /= np.linalg.norm(tfidf, axis=1, keepdims=True)

count_vectorizer = CountVectorizer()
sk_bow = count_vectorizer.fit_transform(documents).toarray()

tfidf_vectorizer = TfidfVectorizer()
sk_tfidf = tfidf_vectorizer.fit_transform(documents).toarray()

print("Vocabulary:", vocabulary)
print("\nBag-of-Words:\n", bow)
print("\nTF-IDF From Scratch:\n", np.round(tfidf, 3))
print("\nScikit-learn Bag-of-Words:\n", sk_bow)
print("\nScikit-learn TF-IDF:\n", np.round(sk_tfidf, 3))

similarity = cosine_similarity(sk_tfidf)

print("\nCosine Similarity:\n", np.round(similarity, 2))

threshold = 0.85

print("\nPlagiarism Results:")
for i in range(n_docs):
    for j in range(i + 1, n_docs):
        score = similarity[i, j]
        status = "PLAGIARISM" if score >= threshold else "NOT PLAGIARISM"
        print(f"Document {i + 1} vs Document {j + 1}: {score:.2f} -> {status}")

features = tfidf_vectorizer.get_feature_names_out()

for i, vector in enumerate(sk_tfidf):
    indices = np.argsort(vector)[-5:][::-1]
    keywords = features[indices]
    scores = vector[indices]

    plt.figure(figsize=(7, 4))
    plt.bar(keywords, scores)
    plt.title(f"Top Keywords - Document {i + 1}")
    plt.xlabel("Keywords")
    plt.ylabel("TF-IDF Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()