"""
Train a naive Bayes classifier from the IMDb reviews data set
"""
from collections import defaultdict
from math import log
from nltk import pos_tag
import re
import os
import random


positive = defaultdict(int)
negative = defaultdict(int)
sums = {'pos': 0, 'neg': 0}

def tokenize(text):
    return re.findall("\w+", text)

def negate_sequence(text):
    """
    Detects negations and transforms negated words into "not_" form.
    """
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    for word in words:
        stripped = word.strip(delims).lower()
        result.append("not_" + stripped if negation else stripped)

        if any(neg in word for neg in frozenset(["not", "n't", "no"])):
            negation = not negation

        if any(c in word for c in delims):
            negation = False
    return result

def get_pos_tags(seq):
    """
    Returns a list of POS tags for word in the sequence, using the Penn Treebank POS Tagger.
    """
    return [tag[1] for tag in pos_tag(seq)]

TWEIGHTS = defaultdict(lambda: 1)
TWEIGHTS.update({
    'JJ': 6, 'RB': 6,
    'JJR': 12, 'RBR': 12,
    'JJS': 18, 'RBS': 18
    })

def tag_weights(text):
    """
    Negate sequence and return weights according to POS tags.
    """
    nseq = negate_sequence(text)
    normalised = [word[4:] if word[:4] == "_not" else word for word in nseq]
    return [(word, TWEIGHTS[tag]) for word, tag in zip(nseq, get_pos_tags(normalised))]

def train():
    for path in random.sample(os.listdir("./aclImdb/train/pos/"), 500):
        path = "./aclImdb/train/pos/" + path
        words = re.findall("\w+", open(path).read().lower())
        for word in words:
            positive[word] += 1
            sums['pos'] += 1
            negative['not_' + word] += 1
            sums['neg'] += 1

    for path in random.sample(os.listdir("./aclImdb/train/neg/"), 500):
        path = "./aclImdb/train/neg/" + path
        words = re.findall("\w+", open(path).read().lower())
        for word in words:
            negative[word] += 1
            sums['neg'] += 1
            positive['not_' + word] += 1
            sums['pos'] += 1

def train_with_weights():
    for path in random.sample(os.listdir("./aclImdb/train/pos/"), 500):
        path = "./aclImdb/train/pos/" + path
        pairs = tag_weights(open(path).read())
        for word, weight in pairs:
            positive[word] += weight
            sums['pos'] += weight

    for path in random.sample(os.listdir("./aclImdb/train/neg/"), 500):
        path = "./aclImdb/train/neg/" + path
        pairs = tag_weights(open(path).read())
        for word, weight in pairs:
            negative[word] += weight
            sums['neg'] += weight

def get_positive_prob(word):
    return 1.0 * (positive[word] + 1) / (2 * sums['pos'])

def get_negative_prob(word):
    return 1.0 * (negative[word] + 1) / (2 * sums['neg'])

def classify(text, pneg = 0.5):
    words = set(negate_sequence(text))
    pscore, nscore = 0, 0

    for word in words:
        pscore += log(get_positive_prob(word))
        nscore += log(get_negative_prob(word))

    return pscore > nscore

def test():
    train()
    strings = [
    "this is really ridiculous",
    ]
    print map(classify, strings)

if __name__ == '__main__':
    print tag_weights("I am feeling good today.")