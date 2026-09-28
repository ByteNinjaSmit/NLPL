
import spacy
from spacy.training import Example
from spacy.scorer import Scorer

# 1. Pretrained NER models
nlp = spacy.load("en_core_web_sm")
text = "John takes Metformin 500 mg for diabetes and nausea."
doc = nlp(text)

print("Pretrained NER:")
for ent in doc.ents:
    print(ent.text, ent.label_)

# 2. Visualize pretrained entities
from spacy import displacy
displacy.serve(doc, style="ent")

# 3. Fine-tune a medical NER model
TRAIN = [
    ("Take Metformin 500 mg for diabetes.",
     [("Metformin", "DRUG"), ("500 mg", "DOSAGE"),
      ("diabetes", "DIAGNOSIS")]),
    ("Patient has fever and headache.",
     [("fever", "SYMPTOM"), ("headache", "SYMPTOM")]),
    ("Prescribed Aspirin 100 mg for heart disease.",
     [("Aspirin", "DRUG"), ("100 mg", "DOSAGE"),
      ("heart disease", "DIAGNOSIS")])
]

ner = spacy.blank("en")
ner.add_pipe("ner")
ner_pipe = ner.get_pipe("ner")

labels = {"DRUG", "DOSAGE", "SYMPTOM", "DIAGNOSIS"}
for label in labels:
    ner_pipe.add_label(label)

examples = []
for sentence, entities in TRAIN:
    doc = ner.make_doc(sentence)
    spans = []
    for word, label in entities:
        start = sentence.index(word)
        spans.append((start, start + len(word), label))
    examples.append(
        Example.from_dict(doc, {"entities": spans})
    )

optimizer = ner.initialize(get_examples=lambda: examples)
for epoch in range(30):
    for example in examples:
        ner.update([example], sgd=optimizer)

# 4. Test and evaluate
test = "Take Aspirin 100 mg for headache."
doc = ner(test)
print("\nMedical NER:")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Evaluate on held-out examples
TEST = [
    ("Take Metformin 500 mg for diabetes.",
     [("Metformin", "DRUG"), ("500 mg", "DOSAGE"),
      ("diabetes", "DIAGNOSIS")])
]

scorer = Scorer()
eval_examples = []
for sentence, entities in TEST:
    predicted = ner(sentence)
    reference = ner.make_doc(sentence)
    spans = []
    for word, label in entities:
        start = sentence.index(word)
        spans.append((start, start + len(word), label))
    eval_examples.append(
        Example.from_dict(predicted, {"entities": spans})
    )

scores = scorer.score(eval_examples)
print("Precision:", scores["ents_p"])
print("Recall:", scores["ents_r"])
print("F1:", scores["ents_f"])