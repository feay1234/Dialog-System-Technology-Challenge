import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel
import collections
import numpy as np
import time
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy


class DST_PICK(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.clf = nn.Linear(config.hidden_size, 3)
        self.binaryCLF = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss_fct = CrossEntropyLoss()
        self.loss_bce = BCELoss()

        self.init_weights()

    def forward(
            self,
            context_tokens=None,
            value_tokens=None,
            slot_labels=None,
            value_labels=None,
    ):
        if context_tokens:
            context_outputs = self.bert(**context_tokens)
            rCLS = context_outputs[0][:, 0, :]
            slot_logits = self.clf(rCLS)

        if value_tokens:
            value_outputs = self.bert(**value_tokens)
            yCLS = value_outputs[0][:, 0, :]
            value_logits = self.sigmoid(self.binaryCLF(yCLS * rCLS))

        if slot_labels is not None and value_labels is not None:
            slot_loss = self.loss_fct(slot_logits, slot_labels.long())
            value_loss = self.loss_bce(value_logits, value_labels.view(-1,1))

            loss = (slot_loss + value_loss) / 2

            return loss.mean()

        if context_tokens and not value_tokens:
            return slot_logits
        else:
            return value_logits

    def generate_train_instances(self, instance, ontology, tokenizer, device):
        gold_slots = ["-".join(g.split("-")[:-1]) for g in instance.gold_state]

        if len(gold_slots) == 0:
            return None

        slots, values, slot_labels, value_labels = [], [], [], []
        for g in instance.gold_state:
            slot = "-".join(g.split("-")[:-1])
            value = g.split("-")[-1]

            # sample by slot
            neg_slot = np.random.choice(list(ontology.keys()))
            while neg_slot in gold_slots:
                neg_slot = np.random.choice(list(ontology.keys()))
            neg_value = np.random.choice(ontology[neg_slot])

            # sample by value
            if slot in ontology:
                cand_values = ontology[slot]
                neg_cand_value = np.random.choice(cand_values)
                while neg_cand_value == value or neg_cand_value == "do n't care":
                    neg_cand_value = np.random.choice(cand_values)
            else:
                neg_cand_value = "none"

            label = 0.0 if value != "dontcare" else 1.0

            slots.extend([slot, neg_slot, slot])
            values.extend([value, neg_value, neg_cand_value])
            slot_labels.extend([label, 2.0, label])
            value_labels.extend([1.0, 0.0, 0.0])

        # Generate inputs
        context_inp = [[slot.replace("-", " "), instance.dialog_history + instance.turn_utter] for slot in slots]
        value_inp = values
        context_tokens = tokenizer(context_inp, padding=True, return_tensors="pt")
        value_tokens = tokenizer(value_inp, padding=True, return_tensors="pt")

        return {"context_tokens": context_tokens.to(device), "value_tokens": value_tokens.to(device),
                "slot_labels": torch.tensor(slot_labels).to(device),
                "value_labels": torch.tensor(value_labels).to(device)}

    def evaluate(self, test_data_raw, tokenizer, ontology, slot_meta, epoch, device):
        slots, values = [], []
        for slot in ontology:
            for value in ontology[slot]:
                slots.append(slot)
                values.append(value)

        op2id = {'update': 0, 'none': 2, 'dontcare': 1}

        id2op = {v: k for k, v in op2id.items()}

        slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
        final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
        op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
        all_op_F1_count = {k: 0 for k in op2id}

        tp_dic = {k: 0 for k in op2id}
        fn_dic = {k: 0 for k in op2id}
        fp_dic = {k: 0 for k in op2id}
        wall_times = []
        results = {}

        # batch_size = 32
        for step in tqdm(range(len(test_data_raw)), desc="Evaluation"):

            instance = test_data_raw[step]

            gold_slot_value = {'-'.join(ii.split('-')[:-1]): ii.split('-')[-1] for ii in instance.gold_state}

            gold_op = []
            pred_op = []
            pred_state = set()
            for j, slot in enumerate(slot_meta):
                if slot not in gold_slot_value:
                    gold_op.append("none")
                else:
                    gold_op.append("update" if gold_slot_value[slot] != "dontcare" else "dontcare")

                start = time.perf_counter()
                # prediction
                context_inp = [[slot.replace("-", " "), instance.dialog_history + instance.turn_utter]]
                context_tokens = tokenizer(context_inp, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.forward(context_tokens)
                    _op = id2op[np.argmax(outputs.cpu().data.numpy())]
                    pred_op.append(_op)

                    if _op == "none":
                        continue
                    elif _op == "dontcare":
                        pred_state.add(slot + "-" + _op)
                    else:
                        slot_pred = []
                        for value in ontology[slot]:
                            value_tokens = tokenizer([value], padding=True, return_tensors="pt").to(device)
                            score = self.forward(context_tokens, value_tokens)
                            slot_pred.append(score)
                        pred_state.add(slot + "-" + ontology[slot][np.argmax(slot_pred)])

                end = time.perf_counter()
                wall_times.append(end - start)

            if set(pred_state) == set(instance.gold_state):
                joint_acc += 1
            key = str(instance.id) + '_' + str(instance.turn_id)

            results[key] = [list(pred_state), instance.gold_state]

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(instance.gold_state), set(pred_state), slot_meta)
            slot_turn_acc += temp_acc

            # Compute prediction F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(instance.gold_state, pred_state)
            slot_F1_pred += temp_f1
            slot_F1_count += count

            # Compute operation accuracy
            temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_op, gold_op)]) / len(pred_op)
            op_acc += temp_acc

            if instance.is_last_turn:
                final_count += 1
                if set(pred_state) == set(instance.gold_state):
                    final_joint_acc += 1
                final_slot_F1_pred += temp_f1
                final_slot_F1_count += count

            # Compute operation F1 score
            for p, g in zip(pred_op, gold_op):
                all_op_F1_count[g] += 1
                if p == g:
                    tp_dic[g] += 1
                    op_F1_count[g] += 1
                else:
                    fn_dic[g] += 1
                    fp_dic[p] += 1
        #
        joint_acc_score = joint_acc / len(test_data_raw)
        turn_acc_score = slot_turn_acc / len(test_data_raw)
        slot_F1_score = slot_F1_pred / slot_F1_count
        op_acc_score = op_acc / len(test_data_raw)
        final_joint_acc_score = final_joint_acc / final_count
        final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
        latency = np.mean(wall_times) * 1000
        op_F1_score = {}
        for k in op2id.keys():
            tp = tp_dic[k]
            fn = fn_dic[k]
            fp = fp_dic[k]
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
            op_F1_score[k] = F1

        # print("------------------------------")
        print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
        print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
        print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
        print("Epoch %d op accuracy : " % epoch, op_acc_score)
        print("Epoch %d op F1 : " % epoch, op_F1_score)
        print("Epoch %d op hit count : " % epoch, op_F1_count)
        print("Epoch %d op all count : " % epoch, all_op_F1_count)
        print("Final Joint Accuracy : ", final_joint_acc_score)
        print("Final slot turn F1 : ", final_slot_F1_score)
        print("Latency Per Prediction : %f ms" % latency)
        print("-----------------------------\n")
        res_per_domain = per_domain_join_accuracy(results, slot_meta)
        #
        scores = {'epoch': epoch, 'joint_acc_score': joint_acc_score,
                  'turn_acc_score': turn_acc_score, 'slot_F1_score': slot_F1_score,
                  'op_acc_score': op_acc_score, 'op_F1_score': op_F1_score, 'op_F1_count': op_F1_count,
                  'all_op_F1_count': all_op_F1_count,
                  'final_joint_acc_score': final_joint_acc_score, 'final_slot_F1_score': final_slot_F1_score,
                  'latency': latency}

        return scores, res_per_domain, results
