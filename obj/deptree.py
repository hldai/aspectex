class DepNode:
    def __init__(self, word):
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
