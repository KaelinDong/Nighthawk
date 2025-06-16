import json
import random
from tqdm import tqdm
from pathlib import Path
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag


MISLEADING_TERMS = ['bug', 'fix', 'error', 'table', 'function', 'mysql', 'functions', 'overflow']

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def targeted_synonym_replace(text, prob=0.3):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    new_words = []

    for word, tag in tagged:
        if word.lower() not in MISLEADING_TERMS:
            new_words.append(word)
            continue

        wn_pos = get_wordnet_pos(tag)
        if wn_pos and random.random() < prob:
            synsets = wn.synsets(word, pos=wn_pos)
            synonyms = {
                lemma.name().replace('_', ' ')
                for syn in synsets
                for lemma in syn.lemmas()
                if lemma.name().lower() != word.lower()
            }
            if synonyms:
                replacement = random.choice(list(synonyms))
                new_words.append(replacement)
                continue
        new_words.append(word)

    return ' '.join(new_words)

def adversarial_augment(jsonl_path, output_path, perturb_prob=0.2):
    input_path = Path(jsonl_path)
    output_path = Path(output_path)
    assert input_path.exists(), f"{input_path} not found."

    with input_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    augmented_data = []
    for item in tqdm(data):
        text = item["commit_message"]
        new_item = item.copy()

        new_item["commit_message"] = targeted_synonym_replace(text, prob=perturb_prob)

        augmented_data.append(new_item)

    with output_path.open("w", encoding="utf-8") as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return f"Adversarial perturbed results: {output_path}"


if __name__ == "__main__":
    adversarial_augment("./dataset/train_dbms_dataset.jsonl", "./dataset/train_dbms_dataset_augmented.jsonl")
