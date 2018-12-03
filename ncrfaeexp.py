import pickle
from models.ncrfae import NeuCRFAutoEncoder
import config
from utils.loggingutils import init_logging
from utils import datautils
import logging
import datetime


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/ncrfae-train-{}.log'.format(str_today), mode='a', to_stdout=True)

    n_tags = 5
    n_train = -1
    label_opinions = True

    # dataset = 'se14l'
    dataset = 'se14r'
    dataset_files = config.DATA_FILES[dataset]

    word_vecs_file = dataset_files['word_vecs_file']
    logging.info('word_vec_file: {}'.format(word_vecs_file))
    logging.info(dataset_files['test_sents_file'])
    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
        # print(vocab)

    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    unlabeled_word_seqs = datautils.read_sents_to_word_idx_seqs(
        dataset_files['unlabeled_tok_sents_file'], word_idx_dict)
    print(len(unlabeled_word_seqs), 'unsupervised sents')

    # n_unlabeled_sents_used = 1000
    n_unlabeled_sents_used = len(unlabeled_word_seqs)
    n_unlabeled_samples_per_iter = 1000
    unsupervised_word_seqs = unlabeled_word_seqs[:n_unlabeled_sents_used]
    logging.info('{} unsupervised sents used.'.format(n_unlabeled_sents_used))

    train_data, valid_data, test_data = datautils.get_data_semeval(
        dataset_files['train_sents_file'], dataset_files['train_tok_texts_file'],
        dataset_files['train_valid_split_file'],
        dataset_files['test_sents_file'], dataset_files['test_tok_texts_file'],
        vocab, n_train, label_opinions)
    ncrfae = NeuCRFAutoEncoder(n_tags, word_vecs_matrix, batch_size=3, lr_method='adam')
    # ncrfae.test_model(train_data)
    ncrfae.train(
        data_train=train_data, data_valid=valid_data, data_test=test_data,
        unlabeled_word_seqs=unlabeled_word_seqs, n_unlabeled_samples_per_iter=n_unlabeled_samples_per_iter,
        n_epochs=500, lr=0.001)
