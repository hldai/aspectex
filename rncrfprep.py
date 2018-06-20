import subprocess

stanford_nlp_lib_file = 'd:/lib/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar'
stanford_nlp_en_parse_file = 'd:/lib/stanford-models/englishPCFG.ser.gz'


def __dependency_parse(text_file, dst_file):
    proc = subprocess.Popen([
        'java', '-cp', stanford_nlp_lib_file,
        'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
        '-nthreads', '2',
        '-sentences', 'newline',
        '-retainTmpSubcategories',
        '-outputFormat', 'typedDependencies',
        '-outputFormatOptions', 'basicDependencies',
        stanford_nlp_en_parse_file,
        text_file],
        stdout=subprocess.PIPE)
    output, err = proc.communicate()

    with open(dst_file, 'wb') as fout:
        fout.write(output)


def __read_sent_dep_tups(fin):
    tups = list()
    for line in fin:
        line = line.strip()
        if not line:
            return tups
        line = line[:-1]
        line = line.replace('(', ' ')
        line = line.replace(', ', ' ')
        tups.append(line.split(' '))
    return tups


def __build_tree_obj(dep_parse_file):

    indice = 0
    plist, rel_list = list(), list()
    vocab = list()
    tree_dict = []

    f = open(dep_parse_file, 'r', encoding='utf-8')
    while True:
        dep_tups = __read_sent_dep_tups(f)
        if not dep_tups:
            break
        print(dep_tups)
    f.close()


text_file = 'd:/data/aspect/rncrf/sample.txt'
dep_parse_file = 'd:/data/aspect/rncrf/raw_parses_sample.txt'
# __dependency_parse(text_file, dep_parse_file)
__build_tree_obj(dep_parse_file)
