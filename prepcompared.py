from utils import utils
import config

tok_texts_file_train = 'd:/projects/ext/Coupled-Multi-layer-Attentions/util/data_semEval/sentence_res15'
tok_texts_file_test = 'd:/projects/ext/Coupled-Multi-layer-Attentions/util/data_semEval/sentence_restest15'
se15_cmla_word_vecs_file = 'd:/data/res/glove-vecs-se15-rest-cmla.txt'
utils.trim_word_vecs_file(
    [tok_texts_file_train, tok_texts_file_test],
    config.GLOVE_WORD_VEC_FILE, se15_cmla_word_vecs_file, 'txt'
)
