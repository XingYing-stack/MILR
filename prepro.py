from tqdm import tqdm
import ujson as json
from collections import defaultdict
import numpy as np

# docred_rel2id = json.load(open('./dataset_docred/meta/rel2id.json', 'r'))
dwie_rel2id = json.load(open('./dataset_dwie/meta/rel2id.json', 'r'))

cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}
# docred_fact_in_train = json.load(open('./dataset_docred/ref/train_annotated.fact', 'r'))
dwie_fact_in_train = json.load(open('./dataset_dwie/ref/train_annotated.fact', 'r'))
# docred_fact_in_train = set(map(lambda x: tuple(x), docred_fact_in_train))
dwie_fact_in_train = set(map(lambda x: tuple(x), dwie_fact_in_train))


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read_dwie(file_in, tokenizer, max_seq_length=1024, Type_Enhance = False):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        if len(entities) <= 1:
            continue
        entity_start, entity_end, type_dir = [], [], {}
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
                type_dir[(sent_id, pos[0])] = mention['type']
                type_dir[(sent_id, pos[1] - 1)] = mention['type']
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    if Type_Enhance:
                        tokens_wordpiece = [type_dir[(i_s, i_t)] + "::start"] + tokens_wordpiece
                    else:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    if Type_Enhance:
                        tokens_wordpiece = tokens_wordpiece + [type_dir[(i_s, i_t) ] + "::end" ]
                    else:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        in_train_dict = defaultdict(bool)

        if "labels" in sample:
            for label in sample['labels']:
                if label['h'] == label['t']:
                    continue
                evidence = label['evidence']
                r = int(dwie_rel2id[label['r']])

                for n1 in sample['vertexSet'][label['h']]:
                    for n2 in sample['vertexSet'][label['t']]:
                        if (n1['name'], n2['name'], label['r']) in dwie_fact_in_train:
                            in_train_dict[(label['h'], label['t'],  r)] = True

                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts, in_trains = [], [], []
        for h, t in train_triple.keys():
            relation = [0] * len(dwie_rel2id)
            in_train = [0] * len(dwie_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                in_train[mention["relation"]] = int(in_train_dict[(h, t, mention["relation"] )])
                evidence = mention["evidence"]
            relations.append(relation)
            in_trains.append(in_train)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    in_train = [0] * len(dwie_rel2id)
                    relation = [1] + [0] * (len(dwie_rel2id) - 1)
                    relations.append(relation)
                    in_trains.append(in_train)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'in_trains': in_trains,
                   'title': sample['title'],
                   }
        features.append(feature)


    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features


def read_docred(file_in, tokenizer, max_seq_length=1024, Type_Enhance=False):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        if len(entities) <= 1:
            continue
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        in_train_dict = defaultdict(bool)
        if "labels" in sample:
            for label in sample['labels']:
                if label['h'] == label['t']:
                    continue
                r = int(docred_rel2id[label['r']])

                for n1 in sample['vertexSet'][label['h']]:
                    for n2 in sample['vertexSet'][label['t']]:
                        if (n1['name'], n2['name'], label['r']) in docred_fact_in_train:
                            in_train_dict[(label['h'], label['t'],  r)] = True

                evidence = label.get('evidence', [])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
        relations, hts, in_trains = [], [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            in_train = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                in_train[mention["relation"]] = int(in_train_dict[(h, t, mention["relation"] )])
                evidence = mention["evidence"]
            relations.append(relation)
            in_trains.append(in_train)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    in_train = [0] * len(docred_rel2id)
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    in_trains.append(in_train)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'in_trains': in_trains,
                   'title': sample['title'],
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features




if __name__ == '__main__':
    with open('./dataset_docred/test.json', 'r') as fh:
        data = json.load(fh)
    print(max([len(d['vertexSet']) * (len(d['vertexSet']) - 1) for d in data]))
