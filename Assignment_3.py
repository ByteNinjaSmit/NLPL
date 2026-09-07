import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
import gensim.downloader as api

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

product_data = [
    ["Wireless Bluetooth Headphones", "wireless bluetooth headphones with noise cancellation deep bass microphone rechargeable battery"],
    ["Bluetooth Earbuds", "true wireless bluetooth earbuds with charging case clear sound deep bass touch controls"],
    ["Gaming Headset", "gaming headset with surround sound noise cancellation microphone RGB lighting comfortable ear cushions"],
    ["Smart Watch", "smart watch with heart rate monitor fitness tracking sleep monitoring GPS AMOLED display"],
    ["Fitness Band", "fitness band with activity tracking heart rate monitor step counter sleep tracking waterproof design"],
    ["Running Shoes", "lightweight running shoes with breathable mesh upper cushioned sole comfortable running footwear"],
    ["Sports Shoes", "sports shoes with durable rubber sole breathable material comfortable athletic footwear"],
    ["Casual Sneakers", "casual sneakers with stylish design lightweight sole comfortable everyday footwear"],
    ["Laptop", "high performance laptop with Intel processor 16GB RAM SSD storage full HD display"],
    ["Gaming Laptop", "gaming laptop with powerful processor dedicated graphics card high refresh rate display 16GB RAM"],
    ["Business Laptop", "business laptop with lightweight design long battery life fast SSD professional keyboard"],
    ["Mechanical Keyboard", "mechanical gaming keyboard with RGB backlight tactile switches USB connection durable keys"],
    ["Wireless Mouse", "wireless mouse with ergonomic design adjustable DPI rechargeable battery smooth tracking"],
    ["Gaming Mouse", "gaming mouse with high precision sensor adjustable DPI RGB lighting programmable buttons"],
    ["Smartphone", "smartphone with AMOLED display powerful processor high resolution camera fast charging"],
    ["Android Phone", "android smartphone with large display powerful processor dual camera fast charging long battery"],
    ["Tablet", "tablet with large touchscreen display powerful processor long battery life lightweight design"],
    ["Power Bank", "portable power bank with high capacity fast charging USB ports compact lightweight design"],
    ["USB Cable", "USB charging cable with fast charging data transfer durable braided design"],
    ["Bluetooth Speaker", "portable bluetooth speaker with powerful sound deep bass waterproof design long battery"],
    ["Smart TV", "smart TV with 4K ultra HD display streaming applications WiFi connectivity Dolby audio"],
    ["LED Monitor", "LED monitor with full HD resolution high refresh rate slim bezel HDMI connectivity"],
    ["4K Monitor", "4K monitor with ultra HD resolution IPS panel high color accuracy HDMI DisplayPort"],
    ["Digital Camera", "digital camera with high resolution sensor optical zoom image stabilization 4K video"],
    ["DSLR Camera", "DSLR camera with interchangeable lens high resolution sensor optical viewfinder professional photography"],
    ["Camera Lens", "camera lens with optical zoom wide aperture image stabilization professional photography"],
    ["Backpack", "water resistant backpack with laptop compartment multiple pockets lightweight durable material"],
    ["Laptop Bag", "laptop bag with padded laptop compartment shoulder strap water resistant material professional design"],
    ["Office Chair", "ergonomic office chair with adjustable height lumbar support comfortable cushion breathable mesh"],
    ["Gaming Chair", "gaming chair with ergonomic backrest adjustable armrests lumbar support reclining design"],
    ["Coffee Maker", "automatic coffee maker with programmable settings fast brewing reusable filter compact design"],
    ["Electric Kettle", "electric kettle with rapid boiling stainless steel body automatic shut off temperature control"],
    ["Air Purifier", "air purifier with HEPA filter removes dust allergens pollutants quiet operation"],
    ["Vacuum Cleaner", "powerful vacuum cleaner with HEPA filter strong suction lightweight design multiple attachments"],
    ["Refrigerator", "energy efficient refrigerator with large storage capacity frost free cooling adjustable shelves"],
    ["Microwave Oven", "microwave oven with multiple cooking modes digital display timer compact kitchen design"],
    ["Smart Bulb", "smart LED bulb with WiFi connectivity adjustable brightness color control mobile application"],
    ["LED Lamp", "LED desk lamp with adjustable brightness flexible neck energy efficient eye protection"],
    ["External SSD", "portable external SSD with high speed data transfer compact design USB connectivity"],
    ["Hard Drive", "external hard drive with large storage capacity USB connectivity reliable data backup"],
    ["Webcam", "full HD webcam with built in microphone autofocus USB connectivity video conferencing"],
    ["Router", "dual band WiFi router with high speed connectivity multiple antennas parental controls"],
    ["WiFi Extender", "WiFi range extender with dual band connectivity improved wireless coverage easy setup"],
    ["Printer", "wireless printer with color printing scanning copying mobile connectivity compact design"],
    ["Monitor Stand", "adjustable monitor stand with ergonomic height adjustment sturdy metal construction"],
    ["Desk", "modern computer desk with spacious surface cable management sturdy metal frame"],
    ["Water Bottle", "stainless steel water bottle insulated design leak proof lid reusable lightweight"],
    ["Travel Mug", "insulated travel mug with stainless steel body leak proof lid hot and cold beverages"],
    ["Yoga Mat", "non slip yoga mat with cushioned surface lightweight waterproof material exercise fitness"],
    ["Dumbbells", "adjustable dumbbells with durable metal construction fitness strength training home workout"]
]

products = pd.DataFrame(product_data, columns=["product", "description"])


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 2]
    return tokens


products["tokens"] = products["description"].apply(preprocess_text)

corpus = products["tokens"].tolist()

word2vec_model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100,
    seed=42
)

word2vec_model.save("ecommerce_word2vec.model")

print("\nWORD2VEC MODEL INFORMATION")
print("Vocabulary Size:", len(word2vec_model.wv))
print("Vector Size:", word2vec_model.vector_size)

query_words = [
    "wireless",
    "gaming",
    "laptop",
    "camera",
    "fitness",
    "battery",
    "display",
    "charging"
]

print("\nWORD2VEC SEMANTIC SIMILARITY")

for word in query_words:
    if word in word2vec_model.wv:
        print("\n", word)
        print(word2vec_model.wv.most_similar(word, topn=5))


def word2vec_analogy(positive, negative):
    available_positive = [
        word for word in positive
        if word in word2vec_model.wv
    ]

    available_negative = [
        word for word in negative
        if word in word2vec_model.wv
    ]

    if not available_positive or not available_negative:
        return []

    return word2vec_model.wv.most_similar(
        positive=available_positive,
        negative=available_negative,
        topn=5
    )


print("\nWORD2VEC ANALOGY")

print(
    "wireless + charging - cable:",
    word2vec_analogy(
        ["wireless", "charging"],
        ["cable"]
    )
)

print(
    "gaming + powerful - office:",
    word2vec_analogy(
        ["gaming", "powerful"],
        ["office"]
    )
)


print("\nLOADING PRE-TRAINED GLOVE MODEL")

glove_model = api.load("glove-wiki-gigaword-100")

print("GloVe Vocabulary Size:", len(glove_model))
print("GloVe Vector Size:", glove_model.vector_size)

print("\nGLOVE SEMANTIC SIMILARITY")

glove_words = [
    "computer",
    "phone",
    "camera",
    "car",
    "king",
    "queen"
]

for word in glove_words:
    if word in glove_model:
        print("\n", word)
        print(glove_model.most_similar(word, topn=5))


print("\nGLOVE ANALOGY")

glove_analogy = glove_model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=10
)

print("king - man + woman")
print(glove_analogy)


print("\nWORD2VEC ANALOGY WITH GENERAL LANGUAGE WORDS")

if all(word in word2vec_model.wv for word in ["king", "man", "woman"]):
    print(
        word2vec_model.wv.most_similar(
            positive=["king", "woman"],
            negative=["man"],
            topn=5
        )
    )
else:
    print("king, man, or woman is not available in the custom e-commerce vocabulary.")


def visualize_pca(model, words, title):
    valid_words = [word for word in words if word in model]

    vectors = np.array([
        model[word]
        for word in valid_words
    ])

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 8))

    plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1]
    )

    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            (
                reduced_vectors[i, 0],
                reduced_vectors[i, 1]
            ),
            fontsize=10
        )

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


visualization_words = [
    "wireless",
    "bluetooth",
    "headphones",
    "earbuds",
    "gaming",
    "laptop",
    "computer",
    "keyboard",
    "mouse",
    "smart",
    "phone",
    "tablet",
    "camera",
    "lens",
    "battery",
    "charging",
    "display",
    "fitness",
    "running",
    "sports",
    "shoes",
    "waterproof",
    "portable",
    "storage",
    "USB",
    "wifi"
]

visualization_words = [
    word.lower()
    for word in visualization_words
]

visualize_pca(
    word2vec_model.wv,
    visualization_words,
    "PCA Visualization of E-Commerce Word2Vec Embeddings"
)


def visualize_tsne(model, words, title):
    valid_words = [word for word in words if word in model]

    vectors = np.array([
        model[word]
        for word in valid_words
    ])

    perplexity = min(10, len(valid_words) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto"
    )

    reduced_vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 8))

    plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1]
    )

    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            (
                reduced_vectors[i, 0],
                reduced_vectors[i, 1]
            ),
            fontsize=10
        )

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


visualize_tsne(
    word2vec_model.wv,
    visualization_words,
    "t-SNE Visualization of E-Commerce Word2Vec Embeddings"
)


product_vectors = []

for tokens in products["tokens"]:
    valid_tokens = [
        token
        for token in tokens
        if token in word2vec_model.wv
    ]

    if valid_tokens:
        vector = np.mean(
            [
                word2vec_model.wv[token]
                for token in valid_tokens
            ],
            axis=0
        )
    else:
        vector = np.zeros(word2vec_model.vector_size)

    product_vectors.append(vector)

product_vectors = np.array(product_vectors)


def recommend_products(query, top_n=5):
    query_tokens = preprocess_text(query)

    valid_tokens = [
        token
        for token in query_tokens
        if token in word2vec_model.wv
    ]

    if not valid_tokens:
        return pd.DataFrame(
            columns=[
                "product",
                "description",
                "similarity"
            ]
        )

    query_vector = np.mean(
        [
            word2vec_model.wv[token]
            for token in valid_tokens
        ],
        axis=0
    ).reshape(1, -1)

    similarities = cosine_similarity(
        query_vector,
        product_vectors
    )[0]

    result = products.copy()

    result["similarity"] = similarities

    result = result.sort_values(
        "similarity",
        ascending=False
    )

    return result[
        [
            "product",
            "description",
            "similarity"
        ]
    ].head(top_n)


search_queries = [
    "wireless headphones",
    "gaming laptop",
    "fitness tracking device",
    "portable charging device",
    "professional camera",
    "comfortable sports footwear",
    "wifi connectivity",
    "home cleaning device"
]

print("\nSEMANTIC PRODUCT RECOMMENDATIONS")

for query in search_queries:
    print("\nSearch Query:", query)

    recommendations = recommend_products(
        query,
        top_n=5
    )

    print(
        recommendations.to_string(
            index=False
        )
    )


def product_similarity(product_name, top_n=5):
    matching_products = products[
        products["product"].str.lower() ==
        product_name.lower()
    ]

    if matching_products.empty:
        return pd.DataFrame()

    index = matching_products.index[0]

    similarities = cosine_similarity(
        product_vectors[index].reshape(1, -1),
        product_vectors
    )[0]

    result = products.copy()
    result["similarity"] = similarities

    result = result.drop(index)

    result = result.sort_values(
        "similarity",
        ascending=False
    )

    return result[
        [
            "product",
            "description",
            "similarity"
        ]
    ].head(top_n)


print("\nPRODUCT-TO-PRODUCT RECOMMENDATIONS")

print(
    product_similarity(
        "Wireless Bluetooth Headphones",
        5
    ).to_string(index=False)
)

print(
    product_similarity(
        "Gaming Laptop",
        5
    ).to_string(index=False)
)

print(
    product_similarity(
        "Digital Camera",
        5
    ).to_string(index=False)
)

print(
    product_similarity(
        "Running Shoes",
        5
    ).to_string(index=False)
)

np.save(
    "ecommerce_product_vectors.npy",
    product_vectors
)

products[
    ["product", "description"]
].to_csv(
    "ecommerce_products.csv",
    index=False
)

print("\nFILES GENERATED")
print("ecommerce_word2vec.model")
print("ecommerce_product_vectors.npy")
print("ecommerce_products.csv")

print("\nSEMANTIC RECOMMENDATION ENGINE READY")