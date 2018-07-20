from os.path import join
from platform import platform

# RAW_WORD_VEC_FILE = 'd:/data/res/GoogleNews-vectors-negative300.bin'
# GOOGLE_NEWS_WORD_VEC_FILE = 'd:/data/res/GoogleNews-vectors-negative300.txt'
# GNEWS_LIGHT_WORD_VEC_FILE = 'd:/data/res/GoogleNews-light-vectors-negative300.txt'

# RAW_WORD_VEC_FILE = 'd:/data/amazon/electronics-vectors.bin'
# WORD_VEC_FILE = 'd:/data/res/amazon-electronics-vectors.txt'
# LIGHT_WORD_VEC_FILE = 'd:/data/res/amazon-electronics-vectors-light.txt'

env = 'Windows' if platform().startswith('Windows') else 'Linux'

if env == 'Windows':
    RES_DIR = 'd:/data/res/'
    DATA_DIR = 'd:/data/aspect/'
    AMAZON_DATA_DIR = 'd:/data/amazon'
else:
    RES_DIR = '/home/data/hldai/res/'
    DATA_DIR = '/home/data/hldai/aspect/'
    AMAZON_DATA_DIR = '/home/data/hldai/amazon'

RAW_WORD_VEC_FILE = join(RES_DIR, 'eletronics-vectors-sg-n10.bin')
WORD_VEC_FILE = join(RES_DIR, 'amazon-eletronics-vectors-sg-n10.txt')
GLOVE_WORD_VEC_FILE = join(RES_DIR, 'glove.6B/glove.6B.100d.txt')

DATA_DIR_HL04 = join(DATA_DIR, 'huliu04')
DATA_DIR_SE14 = join(DATA_DIR, 'semeval14')

# SE14_LAPTOP_GLOVE_WORD_VEC_FILE = 'd:/data/aspect/semeval14/model-data/glove-word-vecs.pkl'
SE14_LAPTOP_GLOVE_WORD_VEC_FILE = join(DATA_DIR_SE14, 'model-data/glove-word-vecs.pkl')

DATA_FILE_LIST_FILE_HL04 = join(DATA_DIR_HL04, 'data-file-names.txt')
REVIEWS_FILE_HL04 = join(DATA_DIR_HL04, 'reviews.json')
SENTS_FILE_HL04 = join(DATA_DIR_HL04, 'sents.json')
SENT_TEXT_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text.txt')
SENT_DEPENDENCY_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text-dep.txt')
SENT_POS_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text-pos.txt')
SEED_ASPECTS_HL04 = join(DATA_DIR_HL04, 'seed-aspects.txt')
SEED_OPINIONS_FILE_HL04 = join(DATA_DIR_HL04, 'opinion-dict.txt')

SE14_LAPTOP_TEST_SENTS_XML_FILE = join(DATA_DIR_SE14, 'Laptops_Test_Gold.xml')
SE14_LAPTOP_TEST_SENTS_FILE = join(DATA_DIR_SE14, 'laptops_test_sents.json')
SE14_LAPTOP_TEST_SENT_TEXTS_FILE = join(DATA_DIR_SE14, 'laptops_test_texts.txt')
SE14_LAPTOP_TEST_TOK_TEXTS_FILE = join(DATA_DIR_SE14, 'laptops_test_texts_tok.txt')
SE14_LAPTOP_TEST_DEP_PARSE_FILE = join(DATA_DIR_SE14, 'laptops_test_dep.txt')
SE14_LAPTOP_TEST_OPINIONS_FILE = join(DATA_DIR_SE14, 'test_laptop_opinions.txt')
SE14_LAPTOP_TRAIN_SENTS_XML_FILE = join(DATA_DIR_SE14, 'Laptops_Train.xml')
SE14_LAPTOP_TRAIN_SENTS_FILE = join(DATA_DIR_SE14, 'laptops_train_sents.json')
SE14_LAPTOP_TRAIN_SENT_TEXTS_FILE = join(DATA_DIR_SE14, 'laptops_train_texts.txt')
SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE = join(DATA_DIR_SE14, 'laptops_train_texts_tok.txt')
SE14_LAPTOP_TRAIN_OPINIONS_FILE = join(DATA_DIR_SE14, 'train_laptop_opinions.txt')
SE14_LAPTOP_TRAIN_DEP_PARSE_FILE = join(DATA_DIR_SE14, 'laptops_train_dep.txt')

SE14_LAPTOP_TEST_RNCRF_DATA_FILE = join(DATA_DIR_SE14, 'laptops_test_rncrf_data.pkl')
SE14_LAPTOP_TRAIN_RNCRF_DATA_FILE = join(DATA_DIR_SE14, 'laptops_train_rncrf_data.pkl')
SE14_LAPTOP_TRAIN_WORD_VECS_FILE = join(DATA_DIR_SE14, 'laptops_train_rncrf_word_vecs.pkl')
SE14_LAPTOP_WORD_VECS_FILE = join(DATA_DIR_SE14, 'laptops_rncrf_word_vecs.pkl')
SE14_LAPTOP_PRE_MODEL_FILE = join(DATA_DIR_SE14, 'laptops_pretrain_model.pkl')
SE14_LAPTOP_MODEL_FILE = join(DATA_DIR_SE14, 'laptops_model.pkl')

AMAZON_TOK_TEXTS_FILE = join(AMAZON_DATA_DIR, 'laptops-reivews-sent-tok-text.txt')
AMAZON_TERMS_TRUE1_FILE = join(AMAZON_DATA_DIR, 'laptops-rule-result1.txt')
AMAZON_TERMS_TRUE2_FILE = join(AMAZON_DATA_DIR, 'laptops-rule-result2.txt')
AMAZON_TERMS_TRUE3_FILE = join(AMAZON_DATA_DIR, 'laptops-rule-result3.txt')
LAPTOP_RULE_MODEL1_FILE = join(DATA_DIR_SE14, 'tf-model/single-1/laptop-rule1.ckpl')
LAPTOP_RULE_MODEL2_FILE = join(DATA_DIR_SE14, 'tf-model/single-2/laptop-rule2.ckpl')
LAPTOP_RULE_MODEL3_FILE = join(DATA_DIR_SE14, 'tf-model/single-3/laptop-rule3.ckpl')
LAPTOP_NRDJ_RULE_MODEL_FILE = join(DATA_DIR_SE14, 'tf-model/laptop-nrdj-rule.ckpl')
