from os.path import join
from platform import platform

# RAW_WORD_VEC_FILE = 'd:/data/res/GoogleNews-vectors-negative300.bin'
# GOOGLE_NEWS_WORD_VEC_FILE = 'd:/data/res/GoogleNews-vectors-negative300.txt'
GNEWS_LIGHT_WORD_VEC_FILE = 'd:/data/res/GoogleNews-light-vectors-negative300.txt'

# RAW_WORD_VEC_FILE = 'd:/data/amazon/electronics-vectors.bin'
# WORD_VEC_FILE = 'd:/data/res/amazon-electronics-vectors.txt'
# LIGHT_WORD_VEC_FILE = 'd:/data/res/amazon-electronics-vectors-light.txt'

env = 'Windows' if platform().startswith('Windows') else 'Linux'

if env == 'Windows':
    BERT_BASE_DIR = 'd:/data/res/bert'
    RES_DIR = 'd:/data/res/'
    DATA_DIR = 'd:/data/aote/'
    AMAZON_DATA_DIR = 'd:/data/amazon'
else:
    BERT_BASE_DIR = '/home/hldai/data/bert'
    RES_DIR = '/home/hldai/data/res/'
    DATA_DIR = '/home/hldai/data/aspect/'
    AMAZON_DATA_DIR = '/home/hldai/data/amazon'

RAW_WORD_VEC_FILE = join(RES_DIR, 'eletronics-vectors-sg-n10.bin')
AMAZON_WORD_VEC_FILE = join(RES_DIR, 'amazon-eletronics-vectors-sg-n10.txt')
GLOVE_WORD_VEC_FILE = join(RES_DIR, 'glove.6B/glove.6B.100d.txt')

DATA_DIR_HL04 = join(DATA_DIR, 'huliu04')
SE14_DIR = join(DATA_DIR, 'semeval14')
SE15_DIR = join(DATA_DIR, 'semeval15')

# SE14_LAPTOP_GLOVE_WORD_VEC_FILE = 'd:/data/aspect/semeval14/model-data/glove-word-vecs.pkl'
SE14_LAPTOP_GLOVE_WORD_VEC_FILE = join(SE14_DIR, 'model-data/laptops-glove-word-vecs.pkl')
SE14_LAPTOP_AMAZON_WORD_VEC_FILE = join(SE14_DIR, 'model-data/laptops-amazon-word-vecs.pkl')

DATA_FILE_LIST_FILE_HL04 = join(DATA_DIR_HL04, 'data-file-names.txt')
REVIEWS_FILE_HL04 = join(DATA_DIR_HL04, 'reviews.json')
SENTS_FILE_HL04 = join(DATA_DIR_HL04, 'sents.json')
SENT_TEXT_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text.txt')
SENT_DEPENDENCY_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text-dep.txt')
SENT_POS_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text-pos.txt')
SEED_ASPECTS_HL04 = join(DATA_DIR_HL04, 'seed-aspects.txt')
SEED_OPINIONS_FILE_HL04 = join(DATA_DIR_HL04, 'opinion-dict.txt')

SE14_LAPTOP_TEST_XML_FILE = join(SE14_DIR, 'laptops/Laptops_Test_Gold.xml')
SE14_LAPTOP_TEST_SENTS_FILE = join(SE14_DIR, 'laptops/laptops_test_sents.json')
SE14_LAPTOP_TEST_SENT_TEXTS_FILE = join(SE14_DIR, 'laptops/laptops_test_texts.txt')
SE14_LAPTOP_TEST_TOK_TEXTS_FILE = join(SE14_DIR, 'laptops/laptops_test_texts_tok_pos.txt')
SE14_LAPTOP_TEST_DEP_PARSE_FILE = join(SE14_DIR, 'laptops/laptops_test_dep.txt')
SE14_LAPTOP_TEST_OPINIONS_FILE = join(SE14_DIR, 'laptops/test_laptop_opinions.txt')
SE14_LAPTOP_TRAIN_XML_FILE = join(SE14_DIR, 'laptops/Laptops_Train.xml')
SE14_LAPTOP_TRAIN_SENTS_FILE = join(SE14_DIR, 'laptops/laptops_train_sents.json')
SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE = join(SE14_DIR, 'laptops/laptops_train_valid_split.txt')
SE14_LAPTOP_TRAIN_SENT_TEXTS_FILE = join(SE14_DIR, 'laptops/laptops_train_texts.txt')
SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE = join(SE14_DIR, 'laptops/laptops_train_texts_tok_pos.txt')
SE14_LAPTOP_TRAIN_OPINIONS_FILE = join(SE14_DIR, 'laptops/train_laptop_opinions.txt')
SE14_LAPTOP_TRAIN_DEP_PARSE_FILE = join(SE14_DIR, 'laptops/laptops_train_dep.txt')

SE14_LAPTOP_TEST_RNCRF_DATA_FILE = join(SE14_DIR, 'laptops_test_rncrf_data.pkl')
SE14_LAPTOP_TRAIN_RNCRF_DATA_FILE = join(SE14_DIR, 'laptops_train_rncrf_data.pkl')
SE14_LAPTOP_TRAIN_WORD_VECS_FILE = join(SE14_DIR, 'laptops_train_rncrf_word_vecs.pkl')
SE14_LAPTOP_WORD_VECS_FILE = join(SE14_DIR, 'model-data/laptops_train_rncrf_word_vecs.pkl')
SE14_LAPTOP_PRE_MODEL_FILE = join(SE14_DIR, 'laptops_pretrain_model.pkl')
SE14_LAPTOP_MODEL_FILE = join(SE14_DIR, 'laptops_model.pkl')

SE14_REST_TRAIN_XML_FILE = join(SE14_DIR, 'restaurants/Restaurants_Train.xml')
SE14_REST_TRAIN_SENTS_FILE = join(SE14_DIR, 'restaurants/restaurants_train_sents.json')
SE14_REST_TRAIN_SENT_TEXTS_FILE = join(SE14_DIR, 'restaurants/restaurants_train_texts.txt')
SE14_REST_TRAIN_TOK_TEXTS_FILE = join(SE14_DIR, 'restaurants/restaurants_train_texts_tok.txt')
SE14_REST_TRAIN_OPINIONS_FILE = join(SE14_DIR, 'restaurants/train_restaurant_opinions.txt')
SE14_REST_TRAIN_DEP_PARSE_FILE = join(SE14_DIR, 'restaurants/restaurants_train_dep.txt')
SE14_REST_TEST_XML_FILE = join(SE14_DIR, 'restaurants/Restaurants_Test_Gold.xml')
SE14_REST_TEST_SENTS_FILE = join(SE14_DIR, 'restaurants/restaurants_test_sents.json')
SE14_REST_TEST_SENT_TEXTS_FILE = join(SE14_DIR, 'restaurants/restaurants_test_texts.txt')
SE14_REST_TEST_TOK_TEXTS_FILE = join(SE14_DIR, 'restaurants/restaurants_test_texts_tok.txt')
SE14_REST_TEST_DEP_PARSE_FILE = join(SE14_DIR, 'restaurants/restaurants_test_dep.txt')
SE14_REST_TEST_OPINIONS_FILE = join(SE14_DIR, 'restaurants/test_restaurant_opinions.txt')
SE14_REST_TRAIN_VALID_SPLIT_FILE = join(SE14_DIR, 'restaurants/restaurants_train_valid_split-6.txt')

SE14_REST_GLOVE_WORD_VEC_FILE = join(SE14_DIR, 'model-data/restaurants_word_vecs.pkl')
SE14_REST_YELP_WORD_VEC_FILE = join(SE14_DIR, 'model-data/restaurants_yelp_word_vecs.pkl')
SE15_REST_YELP_WORD_VEC_FILE = join(SE15_DIR, 'model-data/restaurants_yelp_word_vecs.pkl')

SE15_REST_TRAIN_XML_FILE = join(SE15_DIR, 'restaurants/ABSA-15_Restaurants_Train_Final.xml')
SE15_REST_TRAIN_SENTS_FILE = join(SE15_DIR, 'restaurants/restaurants_train_sents.json')
SE15_REST_TRAIN_SENT_TEXTS_FILE = join(SE15_DIR, 'restaurants/restaurants_train_texts.txt')
SE15_REST_TRAIN_TOK_TEXTS_FILE = join(SE15_DIR, 'restaurants/restaurants_train_texts_tok.txt')
SE15_REST_TRAIN_OPINIONS_FILE = join(SE15_DIR, 'restaurants/opinions_train.txt')
SE15_REST_TRAIN_VALID_SPLIT_FILE = join(SE15_DIR, 'restaurants/restaurants_train_valid_split.txt')

SE15_REST_TEST_XML_FILE = join(SE15_DIR, 'restaurants/ABSA15_Restaurants_Test.xml')
SE15_REST_TEST_SENTS_FILE = join(SE15_DIR, 'restaurants/restaurants_test_sents.json')
SE15_REST_TEST_SENT_TEXTS_FILE = join(SE15_DIR, 'restaurants/restaurants_test_texts.txt')
SE15_REST_TEST_TOK_TEXTS_FILE = join(SE15_DIR, 'restaurants/restaurants_test_texts_tok.txt')
SE15_REST_TEST_OPINIONS_FILE = join(SE15_DIR, 'restaurants/opinions_test.txt')

AMAZON_TOK_TEXTS_FILE = join(AMAZON_DATA_DIR, 'laptops-reivews-sent-tok-text.txt')
AMAZON_TERMS_TRUE1_FILE = join(AMAZON_DATA_DIR, 'laptops-rule-result1.txt')
AMAZON_TERMS_TRUE2_FILE = join(AMAZON_DATA_DIR, 'laptops-rule-result2.txt')
AMAZON_RM_TERMS_FILE = join(AMAZON_DATA_DIR, 'laptops-aspect-rm-rule-result.txt')
# AMAZON_TERMS_TRUE4_FILE = join(AMAZON_DATA_DIR, 'laptops-rule-result4.txt')
AMAZON_TERMS_TRUE4_FILE = join(AMAZON_DATA_DIR, 'laptops-opinion-rule-result.txt')
LAPTOP_RULE_MODEL1_FILE = join(SE14_DIR, 'tf-model/single-1/laptop-rule1.ckpl')
LAPTOP_RULE_MODEL2_FILE = join(SE14_DIR, 'tf-model/single-2/laptop-rule2.ckpl')
LAPTOP_RULE_MODEL3_FILE = join(SE14_DIR, 'tf-model/single-3/laptop-rule3.ckpl')
LAPTOP_RULE_MODEL4_FILE = join(SE14_DIR, 'tf-model/single-3/laptop-rule4.ckpl')
LAPTOP_NRDJ_RULE_MODEL_FILE = join(SE14_DIR, 'tf-model/laptop-nrdj-rule.ckpl')

SE14L_FILES = {
    'train_sents_file': join(SE14_DIR, 'laptops/laptops_train_sents.json'),
    'test_sents_file': join(SE14_DIR, 'laptops/laptops_test_sents.json'),
    'train_valid_split_file': join(SE14_DIR, 'laptops/laptops_train_valid_split.txt'),
    'train_tok_texts_file': join(SE14_DIR, 'laptops/laptops_train_texts_tok_pos.txt'),
    'test_tok_texts_file': join(SE14_DIR, 'laptops/laptops_test_texts_tok_pos.txt'),
    'train_dep_tags_file': join(SE14_DIR, 'laptops/laptops-train-rule-dep.txt'),
    'test_dep_tags_file': join(SE14_DIR, 'laptops/laptops-test-rule-dep.txt'),
    'train_pos_tags_file': join(SE14_DIR, 'laptops/laptops-train-rule-pos.txt'),
    'test_pos_tags_file': join(SE14_DIR, 'laptops/laptops-test-rule-pos.txt'),
    # 'word_vecs_file': join(SE14_DIR, 'model-data/amazon-wv-100-sg-n10-w8-i30.pkl'),
    'word_vecs_file': join(SE14_DIR, 'model-data/laptops-amazon-word-vecs.pkl'),
    'aspect_term_filter_vocab_file': join(SE14_DIR, 'laptops/aspect_filter_vocab_full.txt'),
    'opinion_term_filter_vocab_file': join(SE14_DIR, 'laptops/opinion_filter_vocab_full.txt'),
    'aspect_term_hit_rate_file': join(SE14_DIR, 'laptops/aspect-term-hit-rate.txt'),
    'opinion_term_hit_rate_file': join(SE14_DIR, 'laptops/opinion-term-hit-rate.txt'),
    'aspect_rule_patterns_file': join(SE14_DIR, 'laptops/aspect_mined_rule_patterns.txt'),
    'opinion_rule_patterns_file': join(SE14_DIR, 'laptops/opinion_mined_rule_patterns.txt'),
    'rule_aspect_result_file': join(SE14_DIR, 'laptops/amazon-laptops-aspect-rm-rule-result-tmp.txt'),
    'rule_opinion_result_file': join(SE14_DIR, 'laptops/amazon-laptops-opinion-rule-result.txt'),
    'unlabeled_tok_sents_file': join(RES_DIR, 'amazon/laptops-reivews-sent-tok-text.txt'),
    'train_tfrecord_file': join(SE14_DIR, 'bert-data/se14l-train.tfrecord'),
    'valid_tfrecord_file': join(SE14_DIR, 'bert-data/se14l-valid.tfrecord'),
    'test_tfrecord_file': join(SE14_DIR, 'bert-data/se14l-test.tfrecord'),
    'bert_train_tokens_file': join(SE14_DIR, 'bert-data/se14l-train-tokens.txt'),
    'bert_valid_tokens_file': join(SE14_DIR, 'bert-data/se14l-valid-tokens.txt'),
    'bert_test_tokens_file': join(SE14_DIR, 'bert-data/se14l-test-tokens.txt'),
    'bert_init_checkpoint': join(BERT_BASE_DIR, 'amazon/model.ckpt-10000'),
    'pretrain_aspect_terms_file': join(SE14_DIR, 'laptops/amazon-laptops-aspect-rm-rule-result.txt'),
    'pretrain_opinion_terms_file': join(SE14_DIR, 'laptops/amazon-laptops-opinion-rule-result.txt'),
    'pretrain_train_aspect_tfrec_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-train-aspect.tfrecord'),
    'pretrain_valid_aspect_tfrec_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-valid-aspect.tfrecord'),
    'pretrain_train_opinion_tfrec_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-train-opinion.tfrecord'),
    'pretrain_valid_opinion_tfrec_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-valid-opinion.tfrecord'),
    'pretrain_valid_token_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-valid-tokens.txt'),
}

SE14R_FILES = {
    'train_sents_file': join(SE14_DIR, 'restaurants/restaurants_train_sents.json'),
    'test_sents_file': join(SE14_DIR, 'restaurants/restaurants_test_sents.json'),
    'train_valid_split_file': join(SE14_DIR, 'restaurants/restaurants_train_valid_split.txt'),
    'train_tok_texts_file': join(SE14_DIR, 'restaurants/restaurants_train_texts_tok.txt'),
    'test_tok_texts_file': join(SE14_DIR, 'restaurants/restaurants_test_texts_tok.txt'),
    'train_dep_tags_file': join(SE14_DIR, 'restaurants/restaurants-train-rule-dep.txt'),
    'test_dep_tags_file': join(SE14_DIR, 'restaurants/restaurants-test-rule-dep.txt'),
    'train_pos_tags_file': join(SE14_DIR, 'restaurants/restaurants-train-rule-pos.txt'),
    'test_pos_tags_file': join(SE14_DIR, 'restaurants/restaurants-test-rule-pos.txt'),
    'word_vecs_file': join(SE14_DIR, 'model-data/yelp-w2v-sg-100-n10-i30-w5.pkl'),
    'aspect_term_filter_vocab_file': join(SE14_DIR, 'restaurants/aspect_filter_vocab_full.txt'),
    'opinion_term_filter_vocab_file': join(SE14_DIR, 'restaurants/opinion_filter_vocab_full.txt'),
    'aspect_term_hit_rate_file': join(SE14_DIR, 'restaurants/aspect-term-hit-rate.txt'),
    'opinion_term_hit_rate_file': join(SE14_DIR, 'restaurants/opinion-term-hit-rate.txt'),
    'aspect_rule_patterns_file': join(SE14_DIR, 'restaurants/aspect_mined_rule_patterns.txt'),
    'opinion_rule_patterns_file': join(SE14_DIR, 'restaurants/opinion_mined_rule_patterns.txt'),
    'train_tfrecord_file': join(SE14_DIR, 'bert-data/se14r-train.tfrecord'),
    'valid_tfrecord_file': join(SE14_DIR, 'bert-data/se14r-valid.tfrecord'),
    'test_tfrecord_file': join(SE14_DIR, 'bert-data/se14r-test.tfrecord'),
    'bert_train_tokens_file': join(SE14_DIR, 'bert-data/se14r-train-tokens.txt'),
    'bert_valid_tokens_file': join(SE14_DIR, 'bert-data/se14r-valid-tokens.txt'),
    'bert_test_tokens_file': join(SE14_DIR, 'bert-data/se14r-test-tokens.txt'),
    'bert_init_checkpoint': join(BERT_BASE_DIR, 'yelp/model.ckpt-10000'),
    'unlabeled_tok_sents_file': join(RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-p0_04.txt'),
    'pretrain_aspect_terms_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-aspect-rule-result.txt'),
    'pretrain_opinion_terms_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-opinion-rule-result.txt'),
    'pretrain_train_aspect_tfrec_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-train-aspect.tfrecord'),
    'pretrain_valid_aspect_tfrec_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-valid-aspect.tfrecord'),
    'pretrain_train_opinion_tfrec_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-train-opinion.tfrecord'),
    'pretrain_valid_opinion_tfrec_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-valid-opinion.tfrecord'),
    'pretrain_valid_token_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-valid-tokens.txt'),
}

SE15R_FILES = {
    'train_sents_file': join(SE15_DIR, 'restaurants/restaurants_train_sents.json'),
    'test_sents_file': join(SE15_DIR, 'restaurants/restaurants_test_sents.json'),
    'train_valid_split_file': join(SE15_DIR, 'restaurants/restaurants_train_valid_split.txt'),
    'train_tok_texts_file': join(SE15_DIR, 'restaurants/restaurants_train_texts_tok.txt'),
    'test_tok_texts_file': join(SE15_DIR, 'restaurants/restaurants_test_texts_tok.txt'),
    'train_dep_tags_file': join(SE15_DIR, 'restaurants/restaurants-train-rule-dep.txt'),
    'test_dep_tags_file': join(SE15_DIR, 'restaurants/restaurants-test-rule-dep.txt'),
    'train_pos_tags_file': join(SE15_DIR, 'restaurants/restaurants-train-rule-pos.txt'),
    'test_pos_tags_file': join(SE15_DIR, 'restaurants/restaurants-test-rule-pos.txt'),
    'word_vecs_file': join(SE15_DIR, 'model-data/yelp-w2v-sg-100-n10-i30-w5.pkl'),
    'aspect_term_filter_vocab_file': join(SE15_DIR, 'restaurants/aspect_filter_vocab_full.txt'),
    'opinion_term_filter_vocab_file': join(SE15_DIR, 'restaurants/opinion_filter_vocab_full.txt'),
    'aspect_term_hit_rate_file': join(SE15_DIR, 'restaurants/aspect-term-hit-rate.txt'),
    'opinion_term_hit_rate_file': join(SE15_DIR, 'restaurants/opinion-term-hit-rate.txt'),
    'aspect_rule_patterns_file': join(SE15_DIR, 'restaurants/aspect_mined_rule_patterns.txt'),
    # 'opinion_rule_patterns_file': join(SE15_DIR, 'restaurants/opinion_mined_rule_patterns.txt'),
    'opinion_rule_patterns_file': join(SE14_DIR, 'restaurants/opinion_mined_rule_patterns.txt'),
    # 'opinion_rule_patterns_file': join(SE15_DIR, 'restaurants/opinion_mined_rule_patterns_5_0.6.txt'),
    # 'opinion_rule_patterns_file': join(SE15_DIR, 'restaurants/opinion_mined_rule_patterns_highrecall.txt'),
    'rule_aspect_result_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-aspect-rule-result.txt'),
    # 'rule_opinion_result_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-opinion-rule-result.txt'),
    'rule_opinion_result_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-rule-ot-highrecall.txt'),
    'train_tfrecord_file': join(SE15_DIR, 'bert-data/se15r-train.tfrecord'),
    'valid_tfrecord_file': join(SE15_DIR, 'bert-data/se15r-valid.tfrecord'),
    'test_tfrecord_file': join(SE15_DIR, 'bert-data/se15r-test.tfrecord'),
    'bert_train_tokens_file': join(SE15_DIR, 'bert-data/se15r-train-tokens.txt'),
    'bert_valid_tokens_file': join(SE15_DIR, 'bert-data/se15r-valid-tokens.txt'),
    'bert_test_tokens_file': join(SE15_DIR, 'bert-data/se15r-test-tokens.txt'),
    'bert_init_checkpoint': join(BERT_BASE_DIR, 'yelp/model.ckpt-10000'),
    'unlabeled_tok_sents_file': join(RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-p0_04.txt'),
    'pretrained_bertnrdj_file': join(SE15_DIR, 'model-data/se15r-yelpr9-rest-p0_04-bert.ckpt'),
    'pretrain_aspect_terms_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-aspect-rule-result.txt'),
    # 'pretrain_opinion_terms_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-opinion-rule-result.txt'),
    'pretrain_opinion_terms_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-rule-ot-highrecall.txt'),
    'pretrain_train_aspect_tfrec_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-train-at.tfrecord'),
    'pretrain_valid_aspect_tfrec_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-valid-at.tfrecord'),
    'pretrain_train_opinion_tfrec_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-train-ot-hr.tfrecord'),
    'pretrain_valid_opinion_tfrec_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-valid-ot-hr.tfrecord'),
    'pretrain_valid_token_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-valid-tokens.txt'),
}

DATA_FILES = {
    'se14l': SE14L_FILES,
    'se14r': SE14R_FILES,
    'se15r': SE15R_FILES,
    'restaurants-yelp': {
        # 'sent_texts_file': join(RES_DIR, 'yelp/yelp-review-eng-tok-sents-round-9.txt'),
        # 'dep_tags_file': join(RES_DIR, 'yelp/yelp-review-round-9-dep.txt'),
        # 'pos_tags_file': join(RES_DIR, 'yelp/yelp-review-round-9-pos.txt'),
        'sent_texts_file': join(RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-p0_04.txt'),
        'train_valid_idxs_file': join(RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-p0_04-tvidxs.txt'),
        'dep_tags_file': join(RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-p0_04-dep.txt'),
        'pos_tags_file': join(RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-p0_04-pos.txt'),
    },
    'laptops-amazon': {
        'sent_texts_file': join(RES_DIR, 'amazon/laptops-reivews-sent-tok-text.txt'),
        'train_valid_idxs_file': join(RES_DIR, 'amazon/laptops-reivews-sent-tok-text-tvidxs.txt'),
        'dep_tags_file': join(RES_DIR, 'amazon/laptops-rule-dep.txt'),
        'pos_tags_file': join(RES_DIR, 'amazon/laptops-rule-pos.txt'),
    }
}

BERT_CONFIG_FILE = join(BERT_BASE_DIR, 'uncased_L-12_H-768_A-12/bert_config.json')
BERT_VOCAB_FILE = join(BERT_BASE_DIR, 'uncased_L-12_H-768_A-12/vocab.txt')
BERT_SEQ_LEN = 128
BERT_EMBED_DIM = 3072
