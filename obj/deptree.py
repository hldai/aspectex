class DepNode:
    def __init__(self, word):
        self.label_error = 0
        if word is not None:
            self.word = word
            self.kids = list()
            self.parent = list()
            self.finished = 0
            self.is_word = 1
            self.true_label = 0
            self.ind = -1
        else:
            self.is_word = 0


class DepTree:
    def __init__(self, word_list):
        self.nodes = list()
        for w in word_list:
            self.nodes.append(DepNode(w))

    def get_node(self, ind):
        return self.nodes[ind]

    def get_word_nodes(self):
        return [node for node in self.nodes if node.is_word]

    def add_edge(self, parent, child, rel):
        self.nodes[parent].kids.append((child, rel))
        self.nodes[child].parent.append((parent, rel))

    # return the raw text of the sentence
    def get_words(self):
        return ' '.join([node.word for node in self.get_word_nodes()[1:]])

    def reset_finished(self):
        for node in self.get_word_nodes():
            node.finished = 0

    # one tree's error is the sum of the error at all nodes of the tree
    def error(self):
        sum_err = 0.0
        for node in self.get_word_nodes():
            # sum += node.ans_error
            sum_err += node.label_error

        return sum_err
