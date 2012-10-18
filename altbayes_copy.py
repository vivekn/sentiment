"""
Train a naive Bayes classifier from the IMDb reviews data set 
"""
from collections import defaultdict
from math import log
import re
import os

positive = defaultdict(int)
negative = defaultdict(int)
sums = {'pos': 0, 'neg': 0}


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
        result.append("not_%s" % stripped if negation else stripped)

        for neg in ["not", "n't", "no"]:
            if (neg in word):
                negation = not negation

        for char in delims:
            if char in word:
                negation = False
    return result


def train():
    for path in os.listdir("./aclImdb/train/pos/"):
        path = "./aclImdb/train/pos/" + path
        words = re.findall("\w+", open(path).read())
        for word in words:
            word = word.lower()
            positive[word] += 1
            sums['pos'] += 1
            negative['not_' + word] += 1
            sums['neg'] += 1

    for path in os.listdir("./aclImdb/train/neg/"):
        path = "./aclImdb/train/neg/" + path
        words = re.findall("\w+", open(path).read())
        for word in words:
            word = word.lower()
            negative[word] += 1
            sums['neg'] += 1
            positive['not_' + word] += 1
            sums['pos'] += 1

def get_positive_prob(word):
    return 1.0 * (positive[word] + 1) / (2 * sums['pos'])

def get_negative_prob(word):
    return 1.0 * (negative[word] + 1) / (2 * sums['neg'])

def classify(text, pneg = 0.5):
    assert len(positive) > 0
    assert len(negative) > 0
    words = set(negate_sequence(text))
    pscore, nscore = 0, 0

    for word in words:
        word = word.lower()
        pscore += log(get_positive_prob(word))
        nscore += log(get_negative_prob(word))

    print pscore, nscore
    return pscore > nscore

def test():
    train()
    strings = [
    "this is really ridiculous",
    ]
    print map(classify, strings)

if __name__ == '__main__':
    test()