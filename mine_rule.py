from tqdm import tqdm
import ujson as json

dwie_rel2id = json.load(open('./dataset_dwie/meta/rel2id.json', 'r'))
# docred_rel_info = json.load(open('./dataset_docred/rel_info.json', 'r'))
# docred_rel2id = json.load(open('./dataset_docred/meta/rel2id.json', 'r'))
import torch
import numpy as np
import copy
import dill as pickle


class Rule:
    def __init__(self, target: int, rel_name: str):
        self.body_lst = []
        self.body_lst_NAMES = []
        self.confidence_lst = []  # closed-world-confidence
        self.hc_lst = []  # head-coverage
        self.target = target
        self.relation_name = rel_name

    def append(self, new_body, new_cofidence, new_hc):
        self.body_lst.append(new_body)
        self.confidence_lst.append(new_cofidence)
        self.hc_lst.append(new_hc)


class RuleMiner:
    def __init__(self, file_in="./dataset_dwie/train_annotated.json", max_rule_length=2, minHC=0.008,
                 minC=0.1, minBodyInstance=3, rel2id=dwie_rel2id, device='cuda:0'):  # todo:test hyper parameter
        self.max_rule_length = max_rule_length
        self.minHC = minHC
        self.minC = minC
        self.minBodyInstance = minBodyInstance
        with open(file_in, "r") as fh:
            data = json.load(fh)
        self.data = data

        rel2id = sorted(rel2id.items(), key=lambda kv: (kv[1], kv[0]))
        rel2id = {kv[0]: kv[1] for kv in rel2id}
        self.rel2id = self.add_inverse(rel2id)
        self.device = device
        if 'docred' in file_in:
            self.pid2name = lambda x: docred_rel_info[x] if 'anti' not in x else ('anti_' + docred_rel_info[x.split(
                '_')[
                -1]])
        elif 'dwie' in file_in:
            self.pid2name = lambda x: x
        self.rules = [Rule(id, self.pid2name(rel)) for rel, id in self.rel2id.items() if rel.lower() != 'na']

        id2entity, entity2id, facts = self.transform()
        facts = torch.tensor(facts).to(self.device).int()
        facts = facts[facts[:, 0] != facts[:, -1], :]
        self.id2entity = id2entity
        self.entity2id = entity2id
        self.facts = facts
        self.facts_target = [self.facts[self.facts[:, 1] == target, :] for target in self.rel2id.values() if
                             target != 0]
        self.facts_target_NameAndID = [(rel, id) for rel, id in self.rel2id.items() if id != 0]
        # list(int)
        self.facts_target_size = [fact.size(0) for fact in self.facts_target]

    def add_inverse(self, rel2id):
        anti_dict = {}
        k = 0
        length = len(rel2id)
        for rel, id in rel2id.items():
            if rel.lower() != 'na':
                anti_dict['anti_' + rel] = k + length
                k += 1
        rel2id.update(anti_dict)
        return rel2id

    def transform(self):
        len_relation = len(self.rel2id) // 2
        id2entity = {}
        entity2id = {}  

        facts = set()  
        k = 0
        for doc_id, sample in tqdm(enumerate(self.data), desc="Example"):
            for entityid, entity in enumerate(sample['vertexSet']):  
                id2entity[k] = [str(doc_id) + '_' + str(entityid), set([entity[i]['name'] for i in range(len(entity))])]
                entity2id[str(doc_id) + '_' + str(entityid)] = k
                k += 1
            for label in sample['labels']:
                h = entity2id[str(doc_id) + '_' + str(label['h'])]
                t = entity2id[str(doc_id) + '_' + str(label['t'])]
                r_real = self.rel2id[label['r']]
                r_anti = r_real + len_relation
                facts.add((h, r_real, t))
                facts.add((t, r_anti, h))
        return id2entity, entity2id, sorted(list(facts))

    def calculate(self, facts_1, facts_2):
        if facts_2.size(0) == 0 or facts_2.size(0) == 0:
            return (-1, -1)

        correct = 0
        for i in range(facts_1.size(0)):
            h, *_, t = facts_1[i]
            is_correct = torch.sum((facts_2[:, 0] == h) & (facts_2[:, -1] == t)).item()
            correct += is_correct
            if is_correct > 1:
                raise Exception('Duplicate!')
        return (correct / facts_1.size(0), correct / facts_2.size(0))


    def estimate_rule(self, body: list, target: int) -> tuple:
        facts_specific_relation_in_body = [self.facts[self.facts[:, 1] == relation, :] for relation in body]
        facts_specific_relation_target = self.facts[self.facts[:, 1] == target, :]
        min_facts_num = min(
            [facts.size(0) for facts in facts_specific_relation_in_body] + [facts_specific_relation_target.size(0)])

        if min_facts_num == 0:
            return (-1, -1, 0)
        if len(body) == 1:
            confidence, hc = self.calculate(facts_specific_relation_in_body[0], facts_specific_relation_target)
            return (confidence, hc, facts_specific_relation_in_body[0].size(0))
        elif len(body) == 2:
            hinge_entity = set(facts_specific_relation_in_body[0][:, -1].cpu().numpy()) & set(
                facts_specific_relation_in_body[1][:, 0].cpu().numpy())
            hinge_entity_num = len(hinge_entity)
            if hinge_entity_num <= self.minBodyInstance:
                return (-1, -1, 0)
            facts_body = []
            for hinge in iter(hinge_entity):
                facts_body_first_half = facts_specific_relation_in_body[0][
                                        facts_specific_relation_in_body[0][:, -1] == hinge, :]  #
                facts_body_last_half = facts_specific_relation_in_body[1][
                                       facts_specific_relation_in_body[1][:, 0] == hinge, :]  # 
                for i in range(facts_body_first_half.size(0)):
                    for j in range(facts_body_last_half.size(0)):
                        fact = torch.cat((facts_body_first_half[i][:-1], facts_body_last_half[j][:]), dim=-1)
                        facts_body.append(fact)
            facts_body = torch.stack(facts_body, dim=0).to(facts_specific_relation_target)
            confidence, hc = self.calculate(facts_body, facts_specific_relation_target)
            return (confidence, hc, facts_body.size(0))


        else:
            raise UserWarning('Not yet implement rule mining for body length bigger than 2!')

    def estimate_rule_all_target(self, body: list) -> list:
        facts_specific_relation_in_body = [self.facts[self.facts[:, 1] == relation, :] for relation in body]
        min_facts_num = min([facts.size(0) for facts in facts_specific_relation_in_body])

        if min_facts_num <= self.minBodyInstance:
            return [(-1, -1, 0) for _ in range(len(self.facts_target))]
        if len(body) == 1:
            result_lst = []
            for facts_specific_relation_target in self.facts_target:
                confidence, hc = self.calculate(facts_specific_relation_in_body[0], facts_specific_relation_target)
                result_lst.append((confidence, hc, facts_specific_relation_in_body[0].size(0)))

            return result_lst
        elif len(body) == 2:
            hinge_entity = set(facts_specific_relation_in_body[0][:, -1].cpu().numpy()) & set(
                facts_specific_relation_in_body[1][:, 0].cpu().numpy())
            hinge_entity_num = len(hinge_entity)
            if hinge_entity_num <= self.minBodyInstance:
                return [(-1, -1, 0) for _ in range(len(self.facts_target))]
            facts_body = []
            for hinge in iter(hinge_entity):
                facts_body_first_half = facts_specific_relation_in_body[0][
                                        facts_specific_relation_in_body[0][:, -1] == hinge, :]  # 
                facts_body_last_half = facts_specific_relation_in_body[1][
                                       facts_specific_relation_in_body[1][:, 0] == hinge, :]  #
                for i in range(facts_body_first_half.size(0)):
                    for j in range(facts_body_last_half.size(0)):
                        fact = torch.cat((facts_body_first_half[i][:-1], facts_body_last_half[j][:]), dim=-1)
                        facts_body.append(fact)
            if len(facts_body) <= self.minBodyInstance:
                return [(-1, -1, 0) for _ in range(len(self.facts_target))]
            facts_body = torch.stack(facts_body, dim=0).to(facts_specific_relation_in_body[0])

            result_lst = []
            for facts_specific_relation_target in self.facts_target:
                confidence, hc = self.calculate(facts_body, facts_specific_relation_target)
                result_lst.append((confidence, hc, facts_body.size(0)))
            return result_lst

        else:
            raise UserWarning('Not yet implement rule mining for body length bigger than 2!')


    def mine_rule(self):
        for rel1, rel1_id in tqdm(self.rel2id.items()):
            if rel1.lower() != 'na':
                result_lst = self.estimate_rule_all_target([rel1_id])
                for i in range(len(result_lst)):
                    confidence, hc, _ = result_lst[i]
                    if self.facts_target_NameAndID[i][-1] != rel1_id and confidence > self.minC and hc > self.minHC:
                        self.rules[i].body_lst_NAMES.append([self.pid2name(rel1)])
                        self.rules[i].body_lst.append([rel1_id])
                        self.rules[i].confidence_lst.append(confidence)
                        self.rules[i].hc_lst.append(hc)

        # self.estimate_rule([1,69], 7)

        for rel1, rel1_id in tqdm(self.rel2id.items()):
            for rel2, rel2_id in self.rel2id.items():
                if rel1.lower() != 'na' and rel2.lower() != 'na' and rel1 != rel2:
                    result_lst = self.estimate_rule_all_target([rel1_id, rel2_id])
                    for i in range(len(result_lst)):
                        confidence, hc, _ = result_lst[i]
                        # if self.facts_target_NameAndID[i][-1] != rel1_id and self.facts_target_NameAndID[i][-1] != rel2_id and confidence > self.minC and hc > self.minHC:
                        if confidence > self.minC and hc > self.minHC:
                            self.rules[i].body_lst_NAMES.append([self.pid2name(rel1), self.pid2name(rel2)])
                            self.rules[i].body_lst.append([rel1_id, rel2_id])
                            self.rules[i].confidence_lst.append(confidence)
                            self.rules[i].hc_lst.append(hc)


if __name__ == '__main__':
    # rule_miner = RuleMiner(file_in="./dataset_docred/train_annotated.json", rel2id=docred_rel2id)
    # rule_miner.mine_rule()
    # with open('rule_docred.pl', 'wb') as file:
    #     pickle.dump(rule_miner, file)

    # with open('rule_docred.pl', 'rb') as file:
    #     rule_miner = pickle.load(file)
    #     pass
    # transform_pID2_rel_name(rule_miner)

    with open('rule_docred.pl', 'rb') as file:
        rule_miner = pickle.load(file)
    pass

    # rule_miner.estimate_rule([rule_miner.rel2id['anti_played_by'], rule_miner.rel2id['character_in']], rule_miner.rel2id['plays_in'])
    #
    # rule_miner.estimate_rule([rule_miner.rel2id['in0-x'], rule_miner.rel2id['gpe0']], rule_miner.rel2id['in0'])
