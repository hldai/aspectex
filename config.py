from os.path import join

RAW_WORD_VEC_FILE = 'd:/data/res/GoogleNews-vectors-negative300.bin'
GOOGLE_NEWS_WORD_VEC_FILE = 'd:/data/res/GoogleNews-vectors-negative300.txt'
GNEWS_LIGHT_WORD_VEC_FILE = 'd:/data/res/GoogleNews-light-vectors-negative300.txt'

DATA_DIR = 'd:/data/aspect/'
DATA_DIR_HL04 = 'd:/data/aspect/huliu04'

DATA_FILE_LIST_FILE_HL04 = join(DATA_DIR_HL04, 'data-file-names.txt')
REVIEWS_FILE_HL04 = join(DATA_DIR_HL04, 'reviews.json')
SENTS_FILE_HL04 = join(DATA_DIR_HL04, 'sents.json')
SENT_TEXT_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text.txt')
SENT_DEPENDENCY_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text-dep.txt')
SENT_POS_FILE_HL04 = join(DATA_DIR_HL04, 'sents-text-pos.txt')
SEED_ASPECTS_HL04 = join(DATA_DIR_HL04, 'seed-aspects.txt')
SEED_OPINIONS_FILE_HL04 = join(DATA_DIR_HL04, 'opinion-dict.txt')
