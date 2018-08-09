from utils import utils
from models import rulescommon


class AspectMineHelper:
    def __init__(self, opinion_terms_vocab_file, sents_train):
        self.opinion_terms_vocab = set(utils.read_lines(opinion_terms_vocab_file))
        self.terms_list_train = utils.aspect_terms_list_from_sents(sents_train)

    def get_candidate_terms(self, dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        noun_phrases = rulescommon.get_noun_phrases(words, pos_tag_seq, None)

        verbs = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            if pos_tag in rulescommon.VB_POS_TAGS:
                verbs.append(w)

        return noun_phrases + verbs
