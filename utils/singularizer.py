class Singularizer:
    def __init__(self):
        pass

    @staticmethod
    def singularize(word):
        if word.endswith('ies') and not word.endswith('movies'):
            return word[:-3] + 'y'
        elif word.endswith('shes'):
            return word[:-2]
        elif word.endswith('s'):
            return word[:-1]
        return word
