from bilm.model import BidirectionalLanguageModel


class ElmoLSTMCRF:
    def __init__(self, elmo_bilm: BidirectionalLanguageModel):
        self.elmo_bilm = elmo_bilm
