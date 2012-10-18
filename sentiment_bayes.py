from math import log, exp 
import re

positive, negative = set(), set()

def train():
    global positive, negative
    positive = set(re.findall("\w+", open('positive-words.txt').read()))
    negative = set(re.findall("\w+", open('negative-words.txt').read()))
    pos_copy = frozenset(positive)
    neg_copy = frozenset(negative)
    positive |= set("not_" + word for word in neg_copy)
    negative |= set("not_" + word for word in pos_copy)

def all_caps(word):
    return not any('a' <= c <= 'z' for c in word)

def classify(sentence):
    assert len(positive) > 0
    assert len(negative) > 0
    words = sentence.split()
    seen = set()
    pscore, nscore = 0, 0
    negation = False

    pneg = 1.0 * len(negative) / (len(positive) + len(negative))
    delims = "?.,!:;"

    l = len(positive) + len(negative)
    hit_prob = - log(l)
    miss_prob = - log(2 * l)

    for word in words:
        if word not in seen:
            upscale = 1
            if all_caps(word):
                upscale = 2
            oword = word
            word = word.lower().strip(delims)

            if negation:
                word = "not_" + word
            seen.add(word)

            if word in positive:
                pscore += log(1-pneg) - log(len(positive)) + log(upscale)
                nscore += log(pneg) - log(2 * len(negative))
                # pscore += log(1-pneg) + log(upscale) + hit_prob 
                # nscore += log(pneg) + log(upscale) + miss_prob 
            elif word in negative:
                nscore += log(pneg) - log(len(negative)) + log(upscale)
                pscore += log(1-pneg) - log(2 * len(positive))
                # nscore += log(pneg) + log(upscale) + hit_prob 
                # pscore += log(1-pneg) + log(upscale) + miss_prob 

            else:
                pscore += log(1-pneg) - log(2 * len(positive))
                nscore += log(pneg) - log(2 * len(negative))
                # pscore += log(1-pneg) + log(upscale) + miss_prob 
                # nscore += log(pneg) + log(upscale) + miss_prob 

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
    print map(classify, strings)

if __name__ == '__main__':
    test()