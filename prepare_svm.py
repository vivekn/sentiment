from pretrained import *
from collections import Counter
import os

tmap = dict(zip(positive.keys() + negative.keys(), xrange(len(positive) + len(negative))))

POSITIVE_PATH = "./aclImdb/train/pos/"
NEGATIVE_PATH = "./aclImdb/train/neg/"

POSITIVE_TEST_PATH = "./aclImdb/test/pos/"
NEGATIVE_TEST_PATH = "./aclImdb/test/neg/"
 
def transform(path, cls):
	words = Counter(negate_sequence(open(path).read()))
	return "%s %s\n" % (cls, ' '.join('%d:%f' % (tmap[k], words[k]) for k in words if k in tmap))

def write_file(ofile, pospath, negpath):
	f = open(ofile, "w")
	for fil in os.listdir(pospath):
		f.write(transform(pospath + fil, "+1"))
	for fil in os.listdir(negpath):
		f.write(transform(negpath + fil, "-1"))

if __name__ == '__main__':
	write_file('train.svmdata', POSITIVE_PATH, NEGATIVE_PATH)
	write_file('test.svmdata', POSITIVE_TEST_PATH, NEGATIVE_TEST_PATH)