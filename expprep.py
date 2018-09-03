import numpy as np
import config
from utils import utils


def __write_lines(src_file, dst_file, idxs):
    f = open(src_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8')
    for i, line in enumerate(f):
        if i in idxs:
            fout.write(line)
    fout.close()
    f.close()


def __rand_laptops(n_sents):
    tok_texts_file = config.AMAZON_TOK_TEXTS_FILE
    aspect_terms_file = config.AMAZON_RM_TERMS_FILE
    opinion_terms_file = config.AMAZON_TERMS_TRUE4_FILE

    dst_tok_texts_file = 'd:/data/amazon/rand-laptops/laptops-tok-texts-{}.txt'.format(n_sents)
    dst_at_file = 'd:/data/amazon/rand-laptops/laptops-aspect-terms-{}.txt'.format(n_sents)
    dst_ot_file = 'd:/data/amazon/rand-laptops/laptops-opinion-terms-{}.txt'.format(n_sents)

    tok_texts = utils.read_lines(tok_texts_file)
    n_sents_total = len(tok_texts)
    rand_perm = np.random.permutation(n_sents_total)
    rand_idxs = rand_perm[:n_sents]
    __write_lines(tok_texts_file, dst_tok_texts_file, rand_idxs)
    __write_lines(aspect_terms_file, dst_at_file, rand_idxs)
    __write_lines(opinion_terms_file, dst_ot_file, rand_idxs)


# __rand_laptops(1000)
# __rand_laptops(80000)
# __rand_laptops(40000)
__rand_laptops(120000)
