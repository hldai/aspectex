import json


def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines


def load_json_objs(filename):
    f = open(filename, encoding='utf-8')
    objs = list()
    for line in f:
        objs.append(json.loads(line))
    f.close()
    return objs


def next_sent_pos(fin):
    pos_tags = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        pos_tags.append(line)
    return pos_tags


def next_sent_dependency(fin):
    dep_list = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        dep_tup = line.split(' ')
        dep_list.append(dep_tup)
    return dep_list


def load_dep_tags_list(filename):
    f = open(filename, encoding='utf-8')
    sent_dep_tags_list = list()
    while True:
        dep_tags = next_sent_dependency(f)
        if not dep_tags:
            break
        sent_dep_tags_list.append(dep_tags)
    f.close()
    return sent_dep_tags_list


def load_pos_tags(filename):
    f = open(filename, encoding='utf-8')
    sent_pos_tags_list = list()
    while True:
        sent_pos_tags = next_sent_pos(f)
        if not sent_pos_tags:
            break
        sent_pos_tags_list.append(sent_pos_tags)
    f.close()
    return sent_pos_tags_list
