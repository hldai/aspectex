import pickle


if __name__ == '__main__':
    labeled_input_file = 'd:/data/aspect/rncrf/labeled_input.pkl'
    seed_i = 12

    with open(labeled_input_file, 'rb') as f:
        vocab, rel_list, trees = pickle.load(f)

    trees_train, trees_test = trees[:75], trees[75:]
