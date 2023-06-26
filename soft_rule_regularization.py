import torch
import torch.nn as nn
import torch.nn.functional as F
import dill as pickle
from mine_rule import Rule, RuleMiner
import json
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Union
import cvxpy as cp
import gurobipy
import scipy.sparse as sparse
from matplotlib import pyplot as plt
from losses import ATLoss
import copy
import pandas as pd
from tqdm import tqdm
from time import *

class SRR(nn.Module):
    def __init__(self, rule_path='./mined_rules/rule_dwie.pl', prior_path="./dataset_dwie/train_annotated.json",
                 minC=0.9,
                 only_1_hop=True, temperature=1, temperature_ILP=1, threshold_1_hop=0.05, threshold_2_hop=0.01,
                 rel2id=None, detach_body=False, num_labels=4):
        super().__init__()
        self.only_1_hop = only_1_hop
        self.temperature = temperature
        self.rule_lst = self.load_and_prune(rule_path, minC)
        self.prior = self.Statistical_prior(prior_path, rel2id)
        self.rel2id = self.add_inverse(rel2id)
        self.detach_body = detach_body
        if self.only_1_hop:
            body_head, confidence_lst = [], []
            for rule in self.rule_lst:
                for body, confidence in zip(rule.body_lst, rule.confidence_lst):
                    body_head.append([body[0], rule.target])
                    confidence_lst.append([confidence])
            self.body_head = torch.LongTensor(body_head)
            self.confidence = torch.tensor(confidence_lst)
            self.threshold = threshold_1_hop
        elif self.only_1_hop == False:
            body_head_1_hop, confidence_1_hop = [], []
            body_head_2_hop, confidence_2_hop = [], []
            for rule in self.rule_lst:
                for body, confidence in zip(rule.body_lst, rule.confidence_lst):
                    if len(body) == 1:
                        body_head_1_hop.append([body[0], rule.target])
                        confidence_1_hop.append([confidence])
                    elif len(body) == 2:
                        body_head_2_hop.append([body[0], body[1], rule.target])
                        confidence_2_hop.append([confidence])
            self.body_head_1_hop = torch.LongTensor(body_head_1_hop)
            self.body_head_2_hop = torch.LongTensor(body_head_2_hop)
            self.confidence_1_hop = torch.tensor(confidence_1_hop)
            self.confidence_2_hop = torch.tensor(confidence_2_hop)
            self.threshold = threshold_1_hop
            self.threshold_2_hop = threshold_2_hop
            # for GI speed
            self.eliminate_anti_rule_for_ILP()


        self.temperature_ILP = temperature_ILP
        self.loss_func = ATLoss()
        self.num_labels = num_labels

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

    def load_and_prune(self, rule_path, minC):
        # rule_miner = RuleMiner()
        with open(rule_path, 'rb') as file:
            rule_miner = pickle.load(file)
        rules_lst_before_prune = rule_miner.rules
        rules_lst_after_prune = []
        for rule in rules_lst_before_prune:
            new_rule = Rule(rule.target, rule.relation_name)
            for i in range(len(rule.body_lst)):
                if self.only_1_hop:
                    if rule.confidence_lst[i] >= minC and len(rule.body_lst[i]) == 1:
                        new_rule.confidence_lst.append(rule.confidence_lst[i])
                        new_rule.body_lst.append(rule.body_lst[i])
                        new_rule.body_lst_NAMES.append(rule.body_lst_NAMES[i])
                        new_rule.hc_lst.append(rule.hc_lst[i])
                else:
                    if rule.confidence_lst[i] >= minC:
                        new_rule.confidence_lst.append(rule.confidence_lst[i])
                        new_rule.body_lst.append(rule.body_lst[i])
                        new_rule.body_lst_NAMES.append(rule.body_lst_NAMES[i])
                        new_rule.hc_lst.append(rule.hc_lst[i])
            if len(new_rule.confidence_lst) > 0:
                rules_lst_after_prune.append(new_rule)
        return rules_lst_after_prune


    def transform_inverse_logits(self, logits: torch.Tensor, hts: list) -> torch.Tensor:
        left, right = 0, 0
        logits_no_NA = logits[:, 1:]
        logits_anti = []
        for i in range(len(hts)):
            right = left + len(hts[i])
            for h, t in hts[i]:
                anti_index = hts[i].index([t, h])
                logits_anti.append(logits_no_NA[left + anti_index, :])
            left = right
        logits_anti = torch.stack(logits_anti, dim=0)
        return torch.cat((logits, logits_anti), dim=-1)

    def cal_2_hop_loss(self, log_probs, hts):
        entity_number = list(map(lambda x: int((1 + np.sqrt(1 + 4 * x)) // 2), [len(lst) for lst in hts]))
        max_entity_number = max(entity_number)

        left, right = 0, 0
        index = []
        mask = []
        for i in range(len(hts)):
            right += len(hts[i])
            entity_pair2pos = defaultdict(int)
            for pos, entity_pair in enumerate(hts[i]):
                entity_pair2pos[tuple(entity_pair)] = pos
            entity_number_i = entity_number[i]
            for h, t in hts[i]:
                tmp = []
                for hinge in range(entity_number_i):
                    if hinge != h and hinge != t:
                        tmp.append([entity_pair2pos[(h, hinge)] + left, entity_pair2pos[(hinge, t)] + left])
                tmp = tmp + [[0, 0] for _ in range(max_entity_number - 2 - len(tmp))]
                mask.append((entity_number_i - 2) * [1] + (max_entity_number - entity_number_i) * [0])
                index.append(tmp)

            left = right
        index = torch.tensor(index).to(log_probs.device)
        mask = torch.tensor(mask).to(log_probs.device)

        loss_2_hop = 0
        if 0 in index.shape:
            return loss_2_hop

        for i in range(self.body_head_2_hop.size(0)):
            r1, r2, target = self.body_head_2_hop[i]
            confidence = self.confidence_2_hop[i]
            log_prob_first_half = torch.index_select(log_probs[:, r1], 0, index[:, :, 0].view(-1)).reshape(
                (-1, max_entity_number - 2))

            log_prob_last_half = torch.index_select(log_probs[:, r2], 0, index[:, :, 1].view(-1)).reshape(
                (-1, max_entity_number - 2))
            log_prob_target = log_probs[:, target]
            if not self.detach_body:
                loss = F.relu(
                    log_prob_first_half + log_prob_last_half + torch.log(confidence) - log_prob_target.unsqueeze(
                        -1) - self.threshold_2_hop) * mask
            else:
                loss = F.relu(
                    log_prob_first_half.detach() + log_prob_last_half.detach() + torch.log(
                        confidence) - log_prob_target.unsqueeze(
                        -1) - self.threshold_2_hop) * mask
            loss, _ = loss.max(-1)
            loss_2_hop += loss.mean()

        return loss_2_hop

    def cal_1_hop_loss(self, log_probs, confidence, body_head):
        log_body_prob = torch.index_select(log_probs, 1, body_head[:, 0])
        log_head_prob = torch.index_select(log_probs, 1, body_head[:, 1])
        if not self.detach_body:
            body_minus_head_bigger_than_zero = F.relu(
                log_body_prob + torch.log(confidence.transpose(0, -1)) - log_head_prob - self.threshold)
        else:
            body_minus_head_bigger_than_zero = F.relu(
                log_body_prob + torch.log(confidence.transpose(0, -1)) - log_head_prob - self.threshold)
        return torch.sum(torch.mean(body_minus_head_bigger_than_zero, dim=0))


    def forward(self, logits: torch.Tensor, hts: list = None) -> torch.Tensor:
        logits = self.transform_inverse_logits(logits, hts)
        if self.only_1_hop:
            assert self.body_head is not None and self.confidence is not None
            log_binary_prob = F.logsigmoid(logits / self.temperature)
            self.body_head = self.body_head.to(logits.device)
            self.confidence = self.confidence.to(logits.device)

            return self.cal_1_hop_loss(log_binary_prob, self.confidence, self.body_head)

        elif self.only_1_hop == False:
            assert self.body_head_2_hop is not None and self.body_head_1_hop is not None
            log_binary_prob = F.logsigmoid(logits / self.temperature)
            self.confidence_2_hop = self.confidence_2_hop.to(logits.device)
            self.confidence_1_hop = self.confidence_1_hop.to(logits.device)
            self.body_head_1_hop = self.body_head_1_hop.to(logits.device)
            self.body_head_2_hop = self.body_head_2_hop.to(logits.device)

            loss_2_hop = self.cal_2_hop_loss(log_binary_prob, hts)
            loss_1_hop = self.cal_1_hop_loss(log_binary_prob, self.confidence_1_hop, self.body_head_1_hop)
            return loss_1_hop + loss_2_hop

    def transform_inverse_for_ILP(self, var: Union[np.ndarray, cp.expressions.variable.Variable], hts: list,
                                  entity_pair2pos: defaultdict):
        var_anti = []
        if type(var) == np.ndarray:
            for h, t in hts:
                anti_index = entity_pair2pos[(t, h)]
                var_anti.append(var[anti_index, :])

            var_anti = np.stack(var_anti, axis=0)
            return np.concatenate((var, var_anti), axis=-1)
        else:
            for h, t in hts:
                anti_index = entity_pair2pos[(t, h)]
                var_anti.append(var[anti_index, :])

            var_anti = cp.vstack(var_anti)
            return cp.hstack([var, var_anti])

    def eliminate_anti_rule_for_ILP(self):
        assert self.body_head_1_hop is not None
        assert self.body_head_2_hop is not None

        def get_anti_relid(relid):
            relation_nums = len(self.rel2id) // 2
            if relid > relation_nums:
                return relid - relation_nums
            else:
                return relid + relation_nums

        set_1_hop, set_2_hop = [], []
        confi_1_hop, confi_2_hop = [], []
        for i in range(self.body_head_1_hop.shape[0]):
            h, t = map(lambda x: x.item(), self.body_head_1_hop[i, :])
            if (get_anti_relid(h), get_anti_relid(t)) not in set_1_hop:
                set_1_hop.append((h, t))
                confi_1_hop.append(self.confidence_1_hop[i])
        for i in range(self.body_head_2_hop.shape[0]):
            h1, h2, t = map(lambda x: x.item(), self.body_head_2_hop[i, :])
            if (get_anti_relid(h2), get_anti_relid(h1), get_anti_relid(t)) not in set_2_hop:
                set_2_hop.append((h1, h2, t))
                confi_2_hop.append(self.confidence_2_hop[i])

        self.body_head_1_hop = torch.LongTensor(np.array(list(set_1_hop)))
        self.body_head_2_hop = torch.LongTensor(np.array(list(set_2_hop)))
        self.confidence_1_hop = torch.stack(confi_1_hop, dim=0)
        self.confidence_2_hop = torch.stack(confi_2_hop, dim=0)

    def Statistical_prior(self, path="./dataset_dwie/train_annotated.json", rel2id=None):
        with open(path, 'r') as fb:
            gold_features = json.load(fb)
        # Prevent 0 relations
        prior = np.ones(shape=(len(rel2id) - 1,))

        gold_facts = defaultdict(set)  # key: relation_name value: set( {'title': 'DW_1095032', 'h_idx': 6, 't_idx': 0})

        total_num = len(rel2id) - 1
        for doc in gold_features:
            title = doc['title']
            for label in doc['labels']:
                relation = label["r"]
                # fact = {'title':title, 'h_idx':label['h'], 't_idx':label['t']}
                fact = (title, label['h'], label['t'])
                if label['h'] != label['t']:
                    if fact not in gold_facts[rel2id[relation] - 1]:
                        total_num += 1
                    gold_facts[rel2id[relation] - 1].add(fact)

        for relid, facts in gold_facts.items():
            prior[relid] += len(facts)

        prior = prior / total_num
        return torch.tensor(prior)


    # Integer Linear Programming with constraints
    def ILP_hard(self, neg_log_prob, neg_log_1_minus_prob, hts, entity_number, silver_label_ori):
        cp.settings.ERROR = [cp.settings.USER_LIMIT]
        cp.settings.SOLUTION_PRESENT = [cp.settings.OPTIMAL, cp.settings.OPTIMAL_INACCURATE, cp.settings.SOLVER_ERROR]
        # (h, t) -> pos
        entity_pair2pos = defaultdict(int)
        for pos, entity_pair in enumerate(hts):
            entity_pair2pos[tuple(entity_pair)] = pos

        #  list(list() ) (head, hinge), (hinge, tail) -> (head, tail)
        index = []
        for h, t in hts:
            for hinge in range(entity_number):
                if hinge != h and hinge != t and (h, hinge) in entity_pair2pos and (hinge, t) in entity_pair2pos:
                    index.append([entity_pair2pos[(h, hinge)], entity_pair2pos[(hinge, t)], entity_pair2pos[(h, t)]])
        index = np.array(index)

        pred_variable = cp.Variable((neg_log_prob.shape[0], neg_log_prob.shape[1]), boolean=True)

        silver_label = self.transform_inverse_for_ILP(silver_label_ori.cpu().numpy(), hts, entity_pair2pos)

        transformed_variable = self.transform_inverse_for_ILP(pred_variable, hts, entity_pair2pos)

        cost_neg_prob = cp.sum((cp.multiply(neg_log_prob, pred_variable) + cp.multiply(neg_log_1_minus_prob,
                                                                                       1 - pred_variable)))

        gap = cp.Constant(1)
        TH = cp.Constant(0)
        number_of_constraints = 0

        valid_1_hop_body_mask = silver_label[:, (self.body_head_1_hop[:, 0] - 1).cpu()].astype(bool)
        if True in valid_1_hop_body_mask:
            constraints_1_hop = [(transformed_variable[:,
                                  (self.body_head_1_hop[:, 0] - 1).cpu()] - transformed_variable[:,
                                                                            (self.body_head_1_hop[:, 1] - 1).cpu()])[
                                     valid_1_hop_body_mask] <= TH]
            number_of_constraints += valid_1_hop_body_mask.sum()
        else:
            constraints_1_hop = []
        constraints_2_hop_left_expression = []

        if index.shape[0] > 0:
            # enumerating all rules
            for i in range(self.body_head_2_hop.shape[0]):
                r1, r2, target = map(lambda x: x - 1, self.body_head_2_hop[i].cpu())
                valid_2_hop_body_mask = (silver_label[index[:, 0], r1] * silver_label[index[:, 1], r2]).astype(bool)
                number_of_constraints += valid_2_hop_body_mask.sum()
                if True not in valid_2_hop_body_mask:
                    continue

                var_body_1 = transformed_variable[index[:, 0], r1][valid_2_hop_body_mask]

                var_body_2 = transformed_variable[index[:, 1], r2][valid_2_hop_body_mask]

                var_head = transformed_variable[index[:, -1], target][valid_2_hop_body_mask]

                constraints_2_hop_left_expression.append(var_body_1 + var_body_2 - var_head - gap)
            if len(constraints_2_hop_left_expression) != 0:
                constraints_2_hop_left_expression = cp.hstack(constraints_2_hop_left_expression)
                constraints_2_hop = [constraints_2_hop_left_expression <= TH]
            else:
                constraints_2_hop = []
        else:
            constraints_2_hop = []
        constraints_max_label_num = [cp.sum(pred_variable, axis=1) <= 4]

        prob = cp.Problem(objective=cp.Minimize(cost_neg_prob),
                          constraints=constraints_1_hop + constraints_max_label_num + constraints_2_hop)

        prob.solve(solver=cp.GUROBI, IterationLimit=20000, verbose=False, QCPDual=0, ignore_dpp=True)
        return torch.tensor(pred_variable.value)


    def global_inference(self, logits: Union[torch.Tensor, list], hts: list, threshold=0,
                         use_prior=False, tau=1, k=1):
        if type(logits) == torch.Tensor:
            logits = torch.split(logits, [len(ht) for ht in hts], dim=0)
        preds_after_ILP = []
        preds_ori = []
        for i in range(len(hts)):
            pred_label_ori = self.loss_func.get_label(logits[i], self.num_labels)
            logits_above_NA = logits[i][:, 1:] - logits[i][:, 0].unsqueeze(-1)
            biggest = torch.max(logits_above_NA, dim=-1)[0].cpu().numpy()

            entity_pair2pos = defaultdict(int)
            for pos, entity_pair in enumerate(hts[i]):
                entity_pair2pos[tuple(entity_pair)] = pos

            hts_MASK = biggest >= threshold
            valid_hts = set(map(lambda x: tuple(x), np.array(hts[i])[hts_MASK, :].tolist()))
            anti_postive_hts = set()
            for h, t in valid_hts:
                anti_postive_hts.add((t, h))
                hts_MASK[entity_pair2pos[(t, h)]] = 1

            valid_hts = valid_hts | anti_postive_hts
            del anti_postive_hts
            assert hts_MASK.sum() == len(valid_hts)
            expanded_hts = set()
            entity_number = int((1 + np.sqrt(1 + 4 * len(hts[i]))) // 2)

            for head, hinge in valid_hts:
                for tail in range(entity_number):
                    if hinge != head and hinge != tail and head != tail and (hinge, tail) in valid_hts and (
                    head, tail) not in valid_hts:
                        expanded_hts.add((head, tail))
                        hts_MASK[entity_pair2pos[(head, tail)]] = 1

            assert hts_MASK.sum() == len(expanded_hts) + len(valid_hts)
            del expanded_hts, valid_hts
            if hts_MASK.sum() > 0:
                logits_above_NA = logits_above_NA[hts_MASK, :]
                valid_hts = np.array(hts[i])[hts_MASK, :].tolist()

                neg_log_porbability = -F.logsigmoid(logits_above_NA.float() / self.temperature_ILP)
                neg_log_1_minus_probability = -torch.log(
                    1 - torch.sigmoid(logits_above_NA.float() / self.temperature_ILP) + 1e-5)
                if use_prior:
                    neg_log_porbability = neg_log_porbability
                    neg_log_1_minus_probability = neg_log_1_minus_probability * torch.pow(
                        -torch.log(self.prior.to(neg_log_1_minus_probability)), k)
                neg_log_1_minus_probability = neg_log_1_minus_probability * tau
                pred_label_ILP = self.ILP_hard(neg_log_porbability.cpu().numpy(),
                                               neg_log_1_minus_probability.cpu().numpy(), valid_hts, entity_number,
                                               pred_label_ori[hts_MASK, 1:]).to(pred_label_ori)
                NA_label = (torch.sum(pred_label_ILP, dim=-1, keepdims=True) == 0).to(pred_label_ILP)
                pred_label_ILP = torch.cat([NA_label, pred_label_ILP], dim=-1)
                pred_label_after_ILP = copy.deepcopy(pred_label_ori)
                pred_label_after_ILP[hts_MASK, :] = pred_label_ILP
            else:
                pred_label_after_ILP = pred_label_ori
            preds_after_ILP.append(pred_label_after_ILP)
            preds_ori.append(pred_label_ori)
        preds_after_ILP = torch.cat(preds_after_ILP, dim=0).float()
        preds_ori = torch.cat(preds_ori, dim=0).float()
        return preds_ori, preds_after_ILP


if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    dwie_rel2id = json.load(open('./dataset_dwie/meta/rel2id.json', 'r'))
    dwie_id2rel = {value: key for key, value in dwie_rel2id.items()}
    #
    dataset = "dwie"
    rule_path = './mined_rules/rule_{}.pl'.format(dataset)
    prior_path = "./dataset_{}/train_annotated.json".format(dataset)
    docred_rel2id = json.load(open('./dataset_docred/meta/rel2id.json', 'r'))
    if dataset == 'dwie':
        rel2id = dwie_rel2id
        minC = 0.98
    elif dataset == 'docred':
        rel2id = docred_rel2id
        minC = 0.7
    srr = SRR(rule_path=rule_path, only_1_hop=False, prior_path=prior_path, minC=minC, temperature_ILP=1,
              rel2id=rel2id)
    for rule in srr.rule_lst:
        for conf,body in zip(rule.confidence_lst, rule.body_lst_NAMES):
            if 'anti_' not in rule.relation_name:
                print(str(body),'->' ,rule.relation_name, ':', conf)