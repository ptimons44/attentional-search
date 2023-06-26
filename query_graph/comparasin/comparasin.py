from transformers import pipeline

classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')

def classify_relationship(sent1, sent2, classifier):
    textual_relations = ["contradiction", "entailment", "neutral"]
    return classifier(sent1 + " " + sent2, textual_relations)
