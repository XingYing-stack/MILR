import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
from mine_rule import Rule, RuleMiner
from soft_rule_regularization import SRR
import dill as pickle
import datetime

class DocREModel(nn.Module):
    def __init__(self, config, model,dataset='dwie', emb_size=768, block_size=64, num_labels=-1, minC=0.98, \
                                                                                                   temperature=1, \
                                                                                                          threshold_1_hop=0.05,
                 threshold_2_hop=0.01, lambda_srr=0.01,rel2id=None, use_SRR=False, only_one_hop = False, detach_body = False):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels )

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.use_SRR = use_SRR

        self.lambda_srr = lambda_srr
        rule_path = './mined_rules/rule_{}.pl'.format(dataset)
        prior_path = "./dataset_{}/train_annotated.json".format(dataset)
        self.rule_path, self.prior_path = rule_path, prior_path
        self.srr = SRR(rule_path=rule_path, prior_path=prior_path, minC=minC, temperature=temperature,
                       threshold_1_hop=threshold_1_hop,
                       threshold_2_hop=threshold_2_hop,
                       rel2id=rel2id, only_1_hop=only_one_hop, detach_body=detach_body)

    def set_srr(self, minC, only_one_hop, detach_body, rel2id):
        self.srr = SRR(rule_path=self.rule_path, prior_path=self.prior_path, minC=minC ,
                       rel2id=rel2id, only_1_hop=only_one_hop, detach_body=detach_body)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def add_anti_to_labels(self, labels:torch.Tensor, hts:list) -> torch.Tensor:
        anti_labels = torch.zeros(size=(labels.size(0), labels.size(1) - 1)).to(labels)
        past_entity_pairs = 0
        for hts_one_doc in hts:
            for index, [h, t] in enumerate(hts_one_doc):
                if labels[index + past_entity_pairs, 0] == 1:
                    break
                anti_labels[past_entity_pairs + hts_one_doc.index([t,h])] = labels[index + past_entity_pairs, 1:]

            past_entity_pairs += len(hts_one_doc)
        return torch.cat((labels, anti_labels), dim=1)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                use_ILP = False,
                tau = 0.8,
                k = 0.5,
                output_for_LogiRE=False,
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        if output_for_LogiRE:
            return logits


        if use_ILP:
            # print('tau:{} k:{}'.format(tau, k))
            output = ( self.srr.global_inference(logits=logits,hts=hts, use_prior=True, tau=tau,k=k )[-1],)
        else:
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)


        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss_srr = self.srr(logits.float() , hts)
            loss_cls = self.loss_fnt(logits.float(), labels.float())
            if self.use_SRR:
                loss = loss_cls + self.lambda_srr * loss_srr
            else:
                loss = loss_cls
            loss_dict = {'loss_cls': loss_cls.item(), 'loss_srr': loss_srr.item()}

            output = (loss.to(sequence_output), loss_dict) + output
        return output
