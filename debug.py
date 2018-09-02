# from seqItem import *
# from numpy import *
from utils import utils


# f = open('d:/data/aspect/huliu04/Nokia 6610.txt')
# f = open('d:/data/aspect/huliu04/Nikon coolpix 4300.txt')
# f = open('d:/data/aspect/huliu04/Creative Labs Nomad Jukebox Zen Xtra 40GB.txt')
f = open('d:/data/aspect/huliu04/Creative Labs Nomad Jukebox Zen Xtra 40GB.txt')
cnt = 0
for line in f:
    if '##' not in line:
        continue
    p = line.find('##')
    terms_str = line[:p].strip()
    if not terms_str:
        continue
    cnt += len(terms_str.split(','))
    # print(line[:p])
f.close()
print(cnt)

# sents = utils.load_json_objs('d:/data/aspect/huliu04/sents.json')
# n_sents = [850, 653, 1822, 391, 598]
# n_sents = [n - 11 for n in n_sents]
# ns = 0
# tcnt = 0
# for i in range(len(n_sents)):
#     sents_tmp = sents[ns:n_sents[i] + ns]
#     ns += n_sents[i]
#     if i < 4:
#         continue
#     cnt = 0
#     for s in sents_tmp:
#         print(s['text'])
#         terms = s.get('aspects', list())
#         print(terms)
#         cnt += len(terms)
#     print(cnt)
#     tcnt += cnt
# print(tcnt)
