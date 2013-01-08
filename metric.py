"""
F-Score metrics for testing classifier, also includes functions for data extraction.
Author: Vivek Narayanan
"""
import os

def get_paths():
    """
    Returns supervised paths annotated with their actual labels.
    """
    posfiles = [("./aclImdb/test/pos/" + f, True) for f in os.listdir("./aclImdb/test/pos/")]
    negfiles = [("./aclImdb/test/neg/" + f, False) for f in os.listdir("./aclImdb/test/neg/")]
    return posfiles + negfiles


def fscore(classifier, file_paths):
    tpos, fpos, fneg, tneg = 0, 0, 0, 0
    for path, label in file_paths:
        result = classifier(open(path).read())
        if label and result:
            tpos += 1
        elif label and (not result):
            fneg += 1
        elif (not label) and result:
            fpos += 1
        else:
            tneg += 1
    prec = 1.0 * tpos / (tpos + fpos)
    recall = 1.0 * tpos / (tpos + fneg)
    f1 = 2 * prec * recall / (prec + recall)
    accu = 100.0 * (tpos + tneg) / (tpos+tneg+fpos+fneg)
    # print "True Positives: %d\nFalse Positives: %d\nFalse Negatives: %d\n" % (tpos, fpos, fneg)
    print "Precision: %lf\nRecall: %lf\nAccuracy: %lf" % (prec, recall, accu)

def main():
    from altbayes import classify, train 
    train()
    fscore(classify, get_paths())

if __name__ == '__main__':
    main()
