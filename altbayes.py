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

def train():
    for path in os.listdir("./aclImdb/train/pos/"):
        path = "./aclImdb/train/pos/" + path
        words = re.findall("\w+", open(path).read())
        for word in words:
            positive[word] += 1
            negative['not_' + word] += 1
            sums['pos'] += 1
            sums['neg'] += 1

    for path in os.listdir("./aclImdb/train/neg/"):
        path = "./aclImdb/train/neg/" + path
        words = re.findall("\w+", open(path).read())
        for word in words:
            negative[word] += 1
            positive['not_' + word] += 1
            sums['neg'] += 1
            sums['pos'] += 1

def get_positive_prob(word):
    return 1.0 * (positive[word] + 1) / (2 * sums['pos'])

def get_negative_prob(word):
    return 1.0 * (negative[word] + 1) / (2 * sums['neg'])

def classify(text, pneg = 0.5):
    assert len(positive) > 0
    assert len(negative) > 0
    words = text.split()
    seen = set()
    pscore, nscore = 0, 0
    negation = False

    delims = "?.,!:;"

    for word in words:
        if word not in seen:
            oword = word
            word = word.lower().strip(delims)

            if negation:
                word = "not_" + word
            seen.add(word)

            pscore += log(get_positive_prob(word))
            nscore += log(get_negative_prob(word))

            if (oword == "not" or "n't" in oword or "no" in oword):
                negation = not negation

            for char in delims:
                if char in oword:
                    negation = False
                    break
    return pscore > nscore

def test():
    train()
    strings = [
    "I am feeling good today"
    ]
    print map(analyze, strings)

if __name__ == '__main__':
    test()