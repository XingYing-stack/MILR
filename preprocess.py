from collections import defaultdict
import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import spacy

import en_core_web_lg
nlp = en_core_web_lg.load()

debug = True
# load NeuralCoref and add it to the pipe of SpaCy's model
import neuralcoref
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

# copied from https://github.com/AndrewZhe/Three-Sentences-Are-All-You-Need/blob/main/src/preprocess/make_dataset_doc_path.py
def extract_path(data, keep_sent_order = True):
    sents = data["sents"]
    nodes = [[] for _ in range(len(data['sents']))]
    e2e_sent = defaultdict(dict)

    # create mention's list for each sentence
    for ns_no, ns in enumerate(data['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(ns_no)

    for sent_id in range(len(sents)):
        for n1 in nodes[sent_id]:
            for n2 in nodes[sent_id]:
                if n1 == n2:
                    continue
                if n2 not in e2e_sent[n1]:
                    e2e_sent[n1][n2] = set()
                e2e_sent[n1][n2].add(sent_id)

    # 2-hop Path
    path_two = defaultdict(dict)
    entityNum = len(data['vertexSet'])
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                    continue
                for s1 in e2e_sent[n1][n3]:
                    for s2 in e2e_sent[n3][n2]:
                        if s1 == s2:
                            continue
                        if n2 not in path_two[n1]:
                            path_two[n1][n2] = []
                        cand_sents = [s1, s2]
                        if keep_sent_order == True:
                            cand_sents.sort()
                        path_two[n1][n2].append((cand_sents, n3))

    # 3-hop Path
    path_three = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                    for cand1 in e2e_sent[n1][n3]:
                        for cand2 in path_two[n3][n2]:
                            if cand1 in cand2[0]:
                                continue
                            if cand2[1] == n1:
                                continue
                            if n2 not in path_three[n1]:
                                path_three[n1][n2] = []
                            cand_sents = [cand1] + cand2[0]
                            if keep_sent_order:
                                cand_sents.sort()
                            path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))

    # Consecutive Path
    consecutive = defaultdict(dict)
    for h in range(entityNum):
        for t in range(h + 1, entityNum):
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    gap = abs(n1['sent_id'] - n2['sent_id'])
                    if gap > 2:
                        continue
                    if t not in consecutive[h]:
                        consecutive[h][t] = []
                        consecutive[t][h] = []
                    if n1['sent_id'] < n2['sent_id']:
                        beg, end = n1['sent_id'], n2['sent_id']
                    else:
                        beg, end = n2['sent_id'], n1['sent_id']

                    consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                    consecutive[t][h].append([[i for i in range(beg, end + 1)]])

    # Merge
    merge = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n2 in path_two[n1]:
                merge[n1][n2] = path_two[n1][n2]
            if n2 in path_three[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += path_three[n1][n2]
                else:
                    merge[n1][n2] = path_three[n1][n2]

            if n2 in consecutive[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += consecutive[n1][n2]
                else:
                    merge[n1][n2] = consecutive[n1][n2]

    # Default Path
    for h in range(len(data['vertexSet'])):
        for t in range(len(data['vertexSet'])):
            if h == t:
                continue
            if t in merge[h]:
                continue
            merge[h][t] = []
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    cand_sents = [n1['sent_id'], n2['sent_id']]
                    if keep_sent_order:
                        cand_sents.sort()
                    merge[h][t].append([cand_sents])

    # Remove redundency
    tp_set = set()
    for n1 in merge.keys():
        for n2 in merge[n1].keys():
            hash_set = set()
            new_list = []
            for t in merge[n1][n2]:
                if tuple(t[0]) not in hash_set:
                    hash_set.add(tuple(t[0]))
                    new_list.append(t[0])
            merge[n1][n2] = new_list

    return merge

def add_sents(data):
    # 改变sent编号
    for sample in tqdm(data, desc="改变sent编号！"):
        ans = []
        doc = nlp(' '.join(sample['sents'][0]))
        length = len(doc)
        sents = list(doc.sents)
        flag = 0
        for i in range(len(sents)):
            start , end = sents[i - flag].start, sents[i].end
            if end - start > 5:
                flag = 0
                ans.append(sample['sents'][0][start:end])
            else:
                flag += 1
        if flag != 0:
            start , end = sents[len(sents) - flag].start, sents[len(sents) - 1].end
            ans.append(sample['sents'][0][start:end])
        # 可能分错的实体名字
        author_name = sample['vertexSet'][-1][0]['name'].split(':')[0].split(' ')[0]
        while author_name not in ans[-1]:
            ans[-1] = ans[-2] + ans.pop(-1)

        assert len(sample['sents'][0]) == sum([len(s ) for s in ans])
        sample['sents'] = ans
        # 改变entity位置
        global_pos2sent_id = np.zeros(shape=(length, ))
        global_pos2sent_id.astype(np.int)
        cumsum =[0] +  list(np.cumsum(list(map(lambda x:len(x), ans) )))
        for i in range(len(cumsum) - 1):
            global_pos2sent_id[cumsum[i] : cumsum[i+1]] = i

        for entity in sample['vertexSet']:
            for mention in entity:
                if int(global_pos2sent_id[mention['pos'][0]]) != int(global_pos2sent_id[mention['pos'][1] - 1]):
                    raise Exception('划分句子有问题！')
                # assert int(global_pos2sent_id[mention['pos'][0]]) == int(global_pos2sent_id[mention['pos'][1] - 1])
                mention['sent_id'] = int(global_pos2sent_id[mention['pos'][0]])
                delay = cumsum[mention['sent_id']]
                mention['pos'] = [mention['pos'][0] - delay, mention['pos'][1] - delay]





with open('./dataset_dwie/test.json', 'r') as fp:
    test_dwie = json.load(fp)
if debug:
    test_dwie = test_dwie[:20]
test_dwie = [test_dwie[6]]
add_sents(data=test_dwie)

print('dada')
