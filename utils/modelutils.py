def filter_incorrect_dep_trees(trees):
    idxs_remove = set()
    for ind, tree in enumerate(trees):
        # the tree is empty
        if not tree.get_word_nodes():
            idxs_remove.add(ind)
        elif tree.get_node(0).is_word == 0:
            print(tree.get_words(), ind)
            idxs_remove.add(ind)

    keep_idxs = [idx for idx in range(len(trees)) if idx not in idxs_remove]
    return [t for i, t in enumerate(trees) if i in keep_idxs]
