from collections import namedtuple

TrainDataBert = namedtuple("TrainDataBert", ["label_seqs", "word_embed_seqs"])
ValidDataBert = namedtuple("ValidDataBert", [
    "label_seqs", "word_embed_seqs", "tok_texts", "aspects_true_list", "opinions_true_list"])


def load_train_data_bert(bert_embed_file, tok_texts_file):
    f_bert = open(bert_embed_file, encoding='utf-8')
    f_bert.close()
