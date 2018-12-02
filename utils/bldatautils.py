import numpy as np
from collections import namedtuple
import json
from utils import datautils

TrainDataBert = namedtuple("TrainDataBert", ["label_seqs", "word_embed_seqs"])
ValidDataBert = namedtuple("ValidDataBert", [
    "label_seqs", "word_embed_seqs", "tok_texts", "aspects_true_list", "opinions_true_list"])

ValidDataBertOL = namedtuple("ValidDataBertOL", ["token_seqs", "aspects_true_list", "opinions_true_list"])


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
            layer_feat_tups.sort(key=lambda x: -x[0])
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


def get_valid_data(token_embed_seqs, token_seqs, aspect_terms_list, opinion_terms_list):
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


def load_valid_data_bert(bert_embed_file, sents_file):
    token_seqs, token_embed_seqs = __load_bert_embed_data(bert_embed_file)
    aspect_terms_list, opinion_terms_list = datautils.load_terms_list(sents_file, True)
    return get_valid_data(token_embed_seqs, token_seqs, aspect_terms_list, opinion_terms_list)
    # cnt_miss = 0
    # label_seqs = list()
    # tok_texts = list()
    # for i, (aspect_terms, opinion_terms) in enumerate(zip(aspect_terms_list, opinion_terms_list)):
    #     y = datautils.label_sentence(token_seqs[i], aspect_terms, opinion_terms)
    #     # if len(aspect_terms) - np.count_nonzero(y == 1) > 0:
    #     #     print(token_seqs[i])
    #     #     print(aspect_terms)
    #     #     print()
    #     cnt_miss += len(aspect_terms) - np.count_nonzero(y == 1)
    #     tok_text = ' '.join(token_seqs[i])
    #     label_seqs.append(y)
    #     tok_texts.append(tok_text)
    # print(cnt_miss, 'missed')
    # return ValidDataBert(label_seqs, token_embed_seqs, tok_texts, aspect_terms_list, opinion_terms_list)


def load_test_data_bert_ol(sents_file, bert_tokens_file):
    aspect_terms_list, opinion_terms_list = datautils.load_terms_list(sents_file, True)
    token_seqs = datautils.read_tokens_file(bert_tokens_file)
    return ValidDataBertOL(token_seqs, aspect_terms_list, opinion_terms_list)


def load_train_data_bert_ol(sents_file, train_valid_split_file, valid_bert_tokens_file):
    from utils.utils import read_lines

    aspect_terms_list, opinion_terms_list = datautils.load_terms_list(sents_file, True)

    tvs_line = read_lines(train_valid_split_file)[0]
    tvs_arr = [int(v) for v in tvs_line.split()]

    # token_seqs_train = datautils.read_tokens_file(train_bert_tokens_file)
    token_seqs_valid = datautils.read_tokens_file(valid_bert_tokens_file)

    aspect_terms_list_train, aspect_terms_list_valid = list(), list()
    opinion_terms_list_train, opinion_terms_list_valid = list(), list()

    assert len(tvs_arr) == len(aspect_terms_list)
    for i, tvs_label in enumerate(tvs_arr):
        if tvs_label == 0:
            aspect_terms_list_train.append(aspect_terms_list[i])
            opinion_terms_list_train.append(opinion_terms_list[i])
        else:
            aspect_terms_list_valid.append(aspect_terms_list[i])
            opinion_terms_list_valid.append(opinion_terms_list[i])

    data_valid = ValidDataBertOL(token_seqs_valid, aspect_terms_list_valid, opinion_terms_list_valid)
    return len(aspect_terms_list_train), data_valid


def load_train_data_bert(bert_embed_file, sents_file, train_valid_split_file):
    from utils.utils import read_lines

    token_seqs, token_embed_seqs = __load_bert_embed_data(bert_embed_file)
    aspect_terms_list, opinion_terms_list = datautils.load_terms_list(sents_file, True)

    tvs_line = read_lines(train_valid_split_file)[0]
    tvs_arr = [int(v) for v in tvs_line.split()]

    token_seqs_train, token_seqs_valid = list(), list()
    token_embed_seqs_train, token_embed_seqs_valid = list(), list()
    aspect_terms_list_train, aspect_terms_list_valid = list(), list()
    opinion_terms_list_train, opinion_terms_list_valid = list(), list()

    assert len(tvs_arr) == len(token_seqs)
    for i, tvs_label in enumerate(tvs_arr):
        if tvs_label == 0:
            token_seqs_train.append(token_seqs[i])
            token_embed_seqs_train.append(token_embed_seqs[i])
            aspect_terms_list_train.append(aspect_terms_list[i])
            opinion_terms_list_train.append(opinion_terms_list[i])
        else:
            token_seqs_valid.append(token_seqs[i])
            token_embed_seqs_valid.append(token_embed_seqs[i])
            aspect_terms_list_valid.append(aspect_terms_list[i])
            opinion_terms_list_valid.append(opinion_terms_list[i])

    cnt_miss = 0
    label_seqs_train = list()
    for i, (aspect_terms, opinion_terms) in enumerate(zip(aspect_terms_list_train, opinion_terms_list_train)):
        y = datautils.label_sentence(token_seqs_train[i], aspect_terms, opinion_terms)
        # if len(aspect_terms) - np.count_nonzero(y == 1) > 0:
        #     print(aspect_terms)
        label_seqs_train.append(y)
        cnt_miss += len(aspect_terms) - np.count_nonzero(y == 1)
    print(cnt_miss, 'missed')
    data_train = TrainDataBert(label_seqs_train, token_embed_seqs_train)

    data_valid = get_valid_data(
        token_embed_seqs_valid, token_seqs_valid, aspect_terms_list_valid, opinion_terms_list_valid)
    return data_train, data_valid
