import os
import bilm

# Our small dataset.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create the vocabulary file with all unique tokens and
# the special <S>, </S> tokens (case sensitive).
all_tokens = set(['<S>', '</S>'] + tokenized_question[0])
for context_sentence in tokenized_context:
    for token in context_sentence:
        all_tokens.add(token)
vocab_file = 'd:/data/res/elmo/tests/vocab_small.txt'
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('d:/data/res/elmo/tests', 'fixtures', 'model')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

# Dump the token embeddings to a file. Run this once for your dataset.
token_embedding_file = 'd:/data/res/elmo/tests/elmo_token_embeddings.hdf5'
bilm.model.dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)
