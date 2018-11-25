import numpy as np
from collections import namedtuple
import json
from utils import datautils

TrainDataBert = namedtuple("TrainDataBert", ["label_seqs", "word_embed_seqs"])
ValidDataBert = namedtuple("ValidDataBert", [
    "label_seqs", "word_embed_seqs", "tok_texts", "aspects_true_list", "opinions_true_list"])


def __load_bert_embed_data(bert_embed_file):
    f_bert = open(bert_embed_file, encoding='utf-8')
    token_seqs, token_embed_seqs = list(), list()
    for i, line in enumerate(f_bert):
        # text = next(f_text)
        # words = text.strip().split(' ')

        tok_feats = json.loads(line)['features']
        toks = list()
        token_embeds = list()
        for feat_obj in tok_feats:
            token = feat_obj['token']
            toks.append(token)
            layers = feat_obj['layers']
            layer_feat_tups = list()
            for layer in layers:
                layer_feat_tups.append((layer['index'], layer['values']))
            layer_feat_tups.sort(key=lambda x: x[0])
            full_feat_vec = list()
            for idx, feat_vec in layer_feat_tups:
                full_feat_vec += feat_vec
            # print(token)
            # print(len(full_feat_vec), full_feat_vec)
            token_embeds.append(np.array(full_feat_vec, np.float32))

        token_seqs.append(toks)
        token_embed_seqs.append(token_embeds)
        # if i > 3:
        #     break
    f_bert.close()
    return token_seqs, token_embed_seqs


def load_valid_data_bert(bert_embed_file, sents_file):
    token_seqs, token_embed_seqs = __load_bert_embed_data(bert_embed_file)
    aspect_terms_list, opinion_terms_list = datautils.load_terms_list(sents_file, True)
    cnt_miss = 0
    label_seqs = list()
    tok_texts = list()
    for i, (aspect_terms, opinion_terms) in enumerate(zip(aspect_terms_list, opinion_terms_list)):
        y = datautils.label_sentence(token_seqs[i], aspect_terms, opinion_terms)
        # if len(aspect_terms) - np.count_nonzero(y == 1) > 0:
        #     print(token_seqs[i])
        #     print(aspect_terms)
        #     print()
        cnt_miss += len(aspect_terms) - np.count_nonzero(y == 1)
        tok_text = ' '.join(token_seqs[i])
        label_seqs.append(y)
        tok_texts.append(tok_text)
    print(cnt_miss, 'missed')
    return ValidDataBert(label_seqs, token_embed_seqs, tok_texts, aspect_terms_list, opinion_terms_list)


def load_train_data_bert(bert_embed_file, sents_file):
    token_seqs, token_embed_seqs = __load_bert_embed_data(bert_embed_file)
    aspect_terms_list, opinion_terms_list = datautils.load_terms_list(sents_file, True)

    cnt_miss = 0
    label_seqs = list()
    for i, (aspect_terms, opinion_terms) in enumerate(zip(aspect_terms_list, opinion_terms_list)):
        y = datautils.label_sentence(token_seqs[i], aspect_terms, opinion_terms)
        # if len(aspect_terms) - np.count_nonzero(y == 1) > 0:
        #     print(aspect_terms)
        label_seqs.append(y)
        cnt_miss += len(aspect_terms) - np.count_nonzero(y == 1)
    print(cnt_miss, 'missed')
    return TrainDataBert(label_seqs, token_embed_seqs)
