# from seqItem import *
# from numpy import *
import string
import tensorflow as tf


f = open('d:/data/res/yelp-review-tok-sents-round-9.txt', encoding='utf-8')
cnt = 0
for line in f:
    cnt += 1
f.close()
print(cnt)
