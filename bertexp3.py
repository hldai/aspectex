import os
import logging
import datetime
from models import robert, bertmodel
from models.bertlstmcrf import BertLSTMCRF
from models.bertnrdj import BertNRDJ
from utils.loggingutils import init_logging
from utils import bldatautils, datautils, utils
import config


def __train_bert():
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/bertlstmcrf-{}.log'.format(str_today), mode='a', to_stdout=True)

    # dataset = 'se14r'
    dataset = 'se15r'

    dataset_files = config.DATA_FILES[dataset]
    if dataset == 'se14l':
        bert_embed_file_train = os.path.join(config.SE14_DIR, 'laptops/laptops_train_texts_tok_bert.txt')
        bert_embed_file_test = os.path.join(config.SE14_DIR, 'laptops/laptops_test_texts_tok_bert.txt')
        # dst_aspects_file = 'd:/data/aspect/semeval14/lstmcrf-aspects.txt'
        # dst_opinions_file = 'd:/data/aspect/semeval14/lstmcrf-opinions.txt'
    elif dataset == 'se14r':
        bert_embed_file_train = os.path.join(
            config.SE14_DIR, 'restaurants/restaurants_train_texts_tok_bert.txt')
        bert_embed_file_test = os.path.join(
            config.SE14_DIR, 'restaurants/restaurants_test_texts_tok_bert.txt')
    else:
        bert_embed_file_train = os.path.join(
            config.SE15_DIR, 'restaurants/restaurants_train_texts_tok_bert.txt')
        bert_embed_file_test = os.path.join(
            config.SE15_DIR, 'restaurants/restaurants_test_texts_tok_bert.txt')

    print('loading data ...')
    data_train, data_valid = bldatautils.load_train_data_bert(
        bert_embed_file_train, dataset_files['train_sents_file'], dataset_files['train_valid_split_file'])
    data_test = bldatautils.load_valid_data_bert(bert_embed_file_test, dataset_files['test_sents_file'])
    print('done')

    word_embed_dim = len(data_train.word_embed_seqs[0][0])
    n_tags = 5
    n_epochs = 100
    lr = 0.001

    # with open(word_vecs_file, 'rb') as f:
    #     vocab, word_vecs_matrix = pickle.load(f)

    logging.info(dataset_files['test_sents_file'])
    logging.info('token_embed_dim={}'.format(word_embed_dim))

    save_model_file = None
    lstmcrf = BertLSTMCRF(n_tags, word_embed_dim, hidden_size_lstm=500, batch_size=5)
    lstmcrf.train(data_train, data_valid, data_test, n_epochs=n_epochs, lr=lr)


def __train_bertlstm_ol():
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/bertlstmcrfol-{}.log'.format(str_today), mode='a', to_stdout=True)

    # dataset = 'se14r'
    dataset = 'se15r'
    n_labels = 5

    dataset_files = config.DATA_FILES[dataset]

    n_train, data_valid = bldatautils.load_train_data_bert_ol(
        dataset_files['train_sents_file'], dataset_files['train_valid_split_file'],
        dataset_files['bert_valid_tokens_file'])
    data_test = bldatautils.load_test_data_bert_ol(
        dataset_files['test_sents_file'], dataset_files['bert_test_tokens_file'])

    bert_config = bertmodel.BertConfig.from_json_file(config.BERT_CONFIG_FILE)
    bm = robert.Robert(
        bert_config, n_labels=n_labels, seq_length=config.BERT_SEQ_LEN, is_train=False,
        init_checkpoint=dataset_files['bert_init_checkpoint']
    )

    lstmcrf = BertLSTMCRF(n_labels, config.BERT_EMBED_DIM, hidden_size_lstm=500, batch_size=5)
    lstmcrf.train_ol(
        robert_model=bm, train_tfrec_file=dataset_files['train_tfrecord_file'],
        valid_tfrec_file=dataset_files['valid_tfrecord_file'], test_tfrec_file=dataset_files['test_tfrecord_file'],
        seq_length=config.BERT_SEQ_LEN, n_train=n_train, data_valid=data_valid, data_test=data_test
    )


def __load_terms_list(sample_idxs, terms_list_file):
    all_terms_list = utils.load_json_objs(terms_list_file)
    dst_terms_list = list()
    for idx in sample_idxs:
        dst_terms_list.append(all_terms_list[idx])
    return dst_terms_list


def __pretrain_bertnrdj(dataset, n_labels, seq_length, n_steps, batch_size, load_model_file, dst_model_file):
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/pre-bertnrdj3-{}-{}.log'.format(utils.get_machine_name(), str_today), mode='a', to_stdout=True)

    dataset_files = config.DATA_FILES[dataset]

    print('init robert ...')
    bert_config = bertmodel.BertConfig.from_json_file(config.BERT_CONFIG_FILE)
    robert_model = robert.Robert(
        bert_config, n_labels=n_labels, seq_length=config.BERT_SEQ_LEN, is_train=False,
        init_checkpoint=dataset_files['bert_init_checkpoint']
    )
    print('done')

    yelp_tv_idxs_file = os.path.join(config.RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-p0_04-tvidxs.txt')
    amazon_tv_idxs_file = os.path.join(config.RES_DIR, 'amazon/laptops-reivews-sent-tok-text-tvidxs.txt')
    tv_idxs_file = amazon_tv_idxs_file if dataset == 'se14l' else yelp_tv_idxs_file
    print('loading data ...')
    idxs_train, idxs_valid = datautils.load_train_valid_idxs(tv_idxs_file)
    logging.info('{} valid samples'.format(len(idxs_valid)))
    # idxs_valid = set(idxs_valid)
    valid_aspect_terms_list = __load_terms_list(idxs_valid, dataset_files['pretrain_aspect_terms_file'])
    valid_opinion_terms_list = __load_terms_list(idxs_valid, dataset_files['pretrain_opinion_terms_file'])
    print('done')

    bertnrdj_model = BertNRDJ(
        n_labels, config.BERT_EMBED_DIM, hidden_size_lstm=hidden_size_lstm, batch_size=batch_size,
        model_file=load_model_file
    )
    bertnrdj_model.pretrain(
        robert_model=robert_model, train_aspect_tfrec_file=dataset_files['pretrain_train_aspect_tfrec_file'],
        valid_aspect_tfrec_file=dataset_files['pretrain_valid_aspect_tfrec_file'],
        train_opinion_tfrec_file=dataset_files['pretrain_train_opinion_tfrec_file'],
        valid_opinion_tfrec_file=dataset_files['pretrain_valid_opinion_tfrec_file'],
        valid_tokens_file=dataset_files['pretrain_valid_token_file'], seq_length=seq_length,
        valid_aspect_terms_list=valid_aspect_terms_list, valid_opinion_terms_list=valid_opinion_terms_list,
        n_steps=n_steps, batch_size=batch_size, save_file=dst_model_file
    )


def __train_bertnrdj(dataset, n_labels, batch_size, model_file):
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/bertnrdj3-{}.log'.format(str_today), mode='a', to_stdout=True)

    dataset_files = config.DATA_FILES[dataset]

    n_train, data_valid = bldatautils.load_train_data_bert_ol(
        dataset_files['train_sents_file'], dataset_files['train_valid_split_file'],
        dataset_files['bert_valid_tokens_file'])
    data_test = bldatautils.load_test_data_bert_ol(
        dataset_files['test_sents_file'], dataset_files['bert_test_tokens_file'])

    bert_config = bertmodel.BertConfig.from_json_file(config.BERT_CONFIG_FILE)
    bm = robert.Robert(
        bert_config, n_labels=n_labels, seq_length=config.BERT_SEQ_LEN, is_train=False,
        init_checkpoint=dataset_files['bert_init_checkpoint']
    )

    # model_file = dataset_files['pretrained_bertnrdj_file']
    # model_file = None
    bertnrdj_model = BertNRDJ(
        n_labels, config.BERT_EMBED_DIM, hidden_size_lstm=hidden_size_lstm, batch_size=batch_size,
        model_file=model_file)
    bertnrdj_model.train(
        robert_model=bm, train_tfrec_file=dataset_files['train_tfrecord_file'],
        valid_tfrec_file=dataset_files['valid_tfrecord_file'], test_tfrec_file=dataset_files['test_tfrecord_file'],
        seq_length=config.BERT_SEQ_LEN, n_train=n_train, data_valid=data_valid, data_test=data_test
    )


if __name__ == '__main__':
    # dataset = 'se14l'
    dataset = 'se14r'
    # dataset = 'se15r'
    n_labels = 5
    seq_length = 128
    n_steps = 400000
    batch_size_pretrain = 32
    batch_size_train = 16
    hidden_size_lstm = 200

    if dataset == 'se14r':
        pretrain_load_model_file = os.path.join(
            config.SE14_DIR, 'model-data/se14r-yelpr9-rest-p0_04-bert-200h.ckpt')
        # model_file = None
        model_file = os.path.join(config.SE14_DIR, 'model-data/se14r-yelpr9-rest-p0_04-bert-200h.ckpt')
    elif dataset == 'se15r':
        pretrain_load_model_file = os.path.join(
            config.SE15_DIR, 'model-data/se15r-yelpr9-rest-p0_04-bert-200h-668.ckpt')
        # model_file = None
        model_file = os.path.join(config.SE15_DIR, 'model-data/se15r-yelpr9-rest-p0_04-bert-200h.ckpt')
    else:
        pretrain_load_model_file = os.path.join(config.SE14_DIR, 'model-data/se14l-amazon-200h.ckpt')
        # model_file = None
        model_file = os.path.join(config.SE14_DIR, 'model-data/se14l-amazon-200h.ckpt')

    # __train_bert()
    # __train_bertlstm_ol()
    # __pretrain_bertnrdj(
    #     dataset=dataset, n_labels=n_labels, seq_length=seq_length, n_steps=n_steps,
    #     batch_size=batch_size_pretrain, load_model_file=pretrain_load_model_file, dst_model_file=model_file)
    __train_bertnrdj(dataset=dataset, n_labels=n_labels, batch_size=batch_size_train, model_file=model_file)
