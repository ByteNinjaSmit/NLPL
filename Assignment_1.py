import re
import emoji
import spacy

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import nltk

# Download NLTK resources (only first time)
nltk.download("stopwords")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize Snowball Stemmer
stemmer = SnowballStemmer("english")

# English stopwords
stop_words = set(stopwords.words("english"))


text = """
Sonam Wangchuk’s letter, teargas, talks with Centre: CJP march as it happened
The march comes even as activist Sonam Wangchuk, who was shifted from the protest site by the Delhi Police to the Safdarjung Hospital on Saturday
Written by: Ariba
5 min readNew DelhiUpdated: Jul 20, 2026 02:03 PM IST
CJPAround 10,000 protesters gather near Jantar Mantar, leading the Delhi Police to rush additional reinforcements. (PTI)
Make us preferred source on Google


Print

Amid heightened security by the Delhi Police, the Cockroach Janta Party (CJP) leaders and supporters began their ‘Sansad Chalo March’ on Monday from Jantar Mantar.

The march comes even as activist Sonam Wangchuk, who was shifted from the protest site by the Delhi Police to the Safdarjung Hospital on Saturday, declared his willingness to break his fast, which has now lasted more than 21 days, only under certain conditions.

As of 12:15 pm, the CJP spokespersons Ashutosh Ranka and Saurav Das were on their way to meet Union Minister of Health and Family Welfare J P Nadda after the government reached out for talks early Monday.

Here’s a timeline of what has happened so far:
Delhi Police tightens security across Delhi ahead of the Cockroach Janta Party’s (CJP) proposed “Chalo Sansad” march on Monday. Section 163 of the BNSS is in place in the New Delhi district, as the police declared no permission was granted for the march.
6:39 am: Activist Sonam Wangchuk lays down conditions for ending his hunger strike after the march. Over a post on X, he asserts that he would break his fast only when the Centre takes responsibility for recent failures in the education system, or when political leaders assure him that the issue would be raised in Parliament.
 8:30 am: A senior police officer informs CJP leader Saurav Das that the government is ready to talk and open a fresh line of communication if the march is “contained”. The CJP leaders remain in touch with the authorities.
8:45 am: Over 3,000 protesters gather around Jantar Mantar, while Delhi police seals off the routes leading to Parliament and imposed prohibitory orders. Protesters were seen raising slogans near the protest site.
9:00 am: New Delhi DCP Sachin Sharma informs The Indian Express that the police had attempted to engage with the protesters, but their requests had been declined. He added, “The area is already beyond capacity and could lead to a law and order situation.”
9:20 am: The Delhi Metro Rail Corporation (DMRC) announces the temporary closure of Rajiv Chowk, Janpath, Patel Chowk, Central Secretariat and Seva Teerth metro stations in view of the proposed march.
9:25 am: Strength of the protesters increase to 8,000.
Around 9:30 am: More than 25 minutes ahead of the scheduled march, Delhi Police erected barricades and prevented protesters from leaving the Jantar Mantar protest site, effectively halting the procession towards Parliament.
9:30 am: CJP spokesperson Ashutosh Ranka asserted that the government was facing a lot of pressure, and that itself was a “victory.”
9:50 am: Around 10,000 protesters gather near Jantar Mantar, leading the Delhi Police to rush additional reinforcements. The police directed all 14 districts to immediately send one ACP-rank officer and 100 personnel each to New Delhi district.
10 am: Activist Sonam Wangchuk’s wife, Gitanjali Angmo, reaches Jantar Mantar.
Around 10 am: AAP MP Sanjay Singh, Bhim Army chief and MP Chandrashekhar Azad, advocate Prashant Bhushan, actor Shabana Azmi, and activist Abhijeet Dipke are present at the protest site.
10:20 am: AISA activists Neha, Aameen and Manish ended their hunger strike after fasting for 23 days at the Jantar Mantar protest site.
10:30 am: Scuffle breaks out at the march near Jantar Mantar. Police personnel charges at a section of protesters towards the Kerala House lane. RAF personnel and Delhi Police also used lathis to control the crowd as protesters clashed with security personnel.
10:50 am: Police use mild force near Patel Chowk Metro station to disperse a group of protesters attempting to make their way towards Jantar Mantar, according to PTI.
11 am: Police remove barricades, and allow the march to proceed. Protesters cross SBI Building on Sansad Marg and continue their march towards the Parliament.
11:30 am: CJP founder Abhijeet Dipke ends his hunger strike at the request of activist Sonam Wangchuk. He breaks his fast by drinking juice.
11:33 am: Visuals show a large number of CJP protesters gathering inside Rajiv Chowk Metro station.
11:45 am: Several protesters topple barricades and move towards Patel Chowk area. Authorities have shut the gates of Patel Chowk Metro station amid the gathering.
11:54 am: Protesters raise slogans near Kerala House, ‘Jab jab Modi darta hai, police ko aage karta hai‘. Security personnel intervene to stop them from moving beyond the protest area.
11:55 am: Samajwadi Party MP Dimple Yadav, along with other party MPs, make their way out of the Parliament towards Jantar Mantar in support of the ongoing student protest.
11:59 am: CJP spokesperson Ashutosh Ranka and Saurav Das are on their way to meet Union Health Minister J P Nadda after the government reaches out for talks.
12:02 pm: Dipke and Wangchuk’s wife Gitanjali J Angmo move towards Parliament on a small truck, surrounded by thousands of supporters.
12:30 pm: Protesters reach the Parliament Street Police Station. As the crowd attempts to move forward, Delhi Police uses lathis to push back and disperse the protesters.
12:40 pm: The Cockroach Janta Party (CJP) temporarily halts march after CJP representatives engage in talks with the government. CJP founder Abhijeet Dipke announces that the march would remain on hold while the dialogue is underway.
Around 1 pm: Former J&K Chief Minister and Jammu & Kashmir National Conference (JKNC) President Farooq Abdullah reaches Jantar Mantar.
Meanwhile, the CJP’s “Sansad Chalo” march remains on hold as its representatives continue talks with the Centre, with protesters waiting for the outcome of the discussions.
"""



def clean_text(text):

# Remove common webpage metadata
    metadata_patterns = [
        r"TIMESOFINDIA\.COM",
        r"Comments?",
        r"Share",
        r"\bAA\b",
        r"Text Size",
        r"Small",
        r"Medium",
        r"Large",
        r"\bIST\b",
    ]

    for pattern in metadata_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags (#word -> word)
    text = re.sub(r"#", "", text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


cleaned = clean_text(text)

print("=" * 60)
print("CLEANED TEXT")
print("=" * 60)
print(cleaned)



# spaCy Processing
doc = nlp(cleaned)

print("\n")
print("=" * 60)
print("TOKENIZATION")
print("=" * 60)

tokens = [token.text for token in doc]
print(tokens)



# Stopword Removal

filtered_tokens = []

for token in doc:
    if token.text.lower() not in stop_words and token.is_alpha:
        filtered_tokens.append(token.text)

print("\n")
print("=" * 60)
print("AFTER STOPWORD REMOVAL")
print("=" * 60)

print(filtered_tokens)



# Snowball Stemming

stems = [stemmer.stem(word.lower()) for word in filtered_tokens]

print("\n")
print("=" * 60)
print("SNOWBALL STEMMING")
print("=" * 60)

print(stems)



# Lemmatization

lemmas = [
    token.lemma_
    for token in doc
    if not token.is_stop
    and not token.is_punct
    and not token.is_space
]

print("\n")
print("=" * 60)
print("LEMMATIZATION (spaCy)")
print("=" * 60)

print(lemmas)



# POS Tagging

print("\n")
print("=" * 80)
print("POS TAGGING")
print("=" * 80)

print(f'{"WORD":20} {"LEMMA":20} {"POS":15}')

for token in doc:
    if token.is_alpha:
        print(f'{token.text:20} {token.lemma_:20} {token.pos_:15}')



# Final Clean Corpus

final_corpus = " ".join(lemmas)

print("\n")
print("=" * 60)
print("FINAL CLEAN CORPUS")
print("=" * 60)

print(final_corpus)