import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel
import collections
import numpy as np
import time
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy


class DST_SPAN(BertPreTrainedModel):
    def __init__(self, config, mode="span"):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.clf = nn.Linear(config.hidden_size, 3)
        self.binaryCLF = nn.Linear(config.hidden_size, 1)

        self.init_weights()

        self.mode = mode

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None,
            span_mask=None,
            slot_label=None,
            value_label=None,
            value_input_ids=None,
            value_attention_mask=None,
            value_token_type_ids=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        rCLS = sequence_output[:, 0, :]

        slot_logits = self.clf(rCLS)

        if value_input_ids:
            value_outputs = self.bert(
                value_input_ids,
                attention_mask=value_attention_mask,
                token_type_ids=value_token_type_ids,
            )
            vCLS = value_outputs[0][:, 0, :]
            sigmoid = nn.Sigmoid()
            value_logits = sigmoid(self.binaryCLF(vCLS))

        outputs = (slot_logits, start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            loss_bce = BCELoss()

            start_loss = loss_fct(start_logits, start_positions) * span_mask
            end_loss = loss_fct(end_logits, end_positions) * span_mask
            slot_loss = loss_fct(slot_logits, slot_label)

            if value_input_ids:
                value_loss = loss_bce(value_logits, value_label.float())
                total_loss = (start_loss + end_loss + slot_loss + value_loss) / 4
            else:
                total_loss = (start_loss + end_loss + slot_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs


def preprocess(tokenizer, context, slot=None, value=None, max_len=128):
    # max_len = 128
    tokenized_context = tokenizer(context)
    if slot:
        tokenized_slot = tokenizer(slot)
    if value:
        tokenized_value = tokenizer(value)['input_ids'][1:-1]

    # Create inputs
    if slot:
        input_ids = tokenized_slot['input_ids'] + tokenized_context['input_ids'][1:]
        token_type_ids = [0] * len(tokenized_slot['input_ids']) + [1] * len(
            tokenized_context['input_ids'][1:]
        )
    else:
        input_ids = tokenized_context['input_ids']
        token_type_ids = [0] * len(tokenized_context['input_ids'])

    attention_mask = [1] * len(input_ids)

    # Pad and create attention masks.
    # Skip if truncation is needed
    padding_length = max_len - len(input_ids)
    if padding_length > 0:  # pad
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

    start, end = -1, -1
    # Find start and stop position of span
    if value:
        l = len(tokenized_value)
        #       search span from the most recent utterance
        for i in range(len(input_ids), l, -1):
            if input_ids[i - l: i] == [0] * l:
                continue
            elif input_ids[i - l: i] == tokenized_value:
                start = i - l
                end = i - 1
                break
    return input_ids, token_type_ids, attention_mask, start, end


def generate_train_data(train_data_raw, ontology, tokenizer, model):
    data = collections.defaultdict(list)

    for idx, instance in enumerate(train_data_raw):

        gold_slots = set(["-".join(g.split("-")[:-1]) for g in instance.gold_state])
        for g_state in instance.gold_state:
            tmp = g_state.split("-")
            slot = "-".join(tmp[:-1])
            value = tmp[-1]

            if slot not in ontology:
                continue

            neg_slot = np.random.choice(list(ontology.keys()))
            while slot == neg_slot or neg_slot in gold_slots:
                neg_slot = np.random.choice(list(ontology.keys()))

            data['slot_label'].append(0 if value != "dontcare" else 1)
            data['slot_label'].append(2)

            #   picklist case
            if "pick" in model.mode:
                cand_values = ontology[slot]
                neg_value = np.random.choice(cand_values)
                while neg_value == value or neg_value == "do n't care":
                    neg_value = np.random.choice(cand_values)

                for val, label in [(value,1.0), (neg_value, 0.0)]:
                    picklist_input_id, picklist_type_id, picklist_mask_id, _, _  = preprocess(tokenizer, val)
                    data['value_input_ids'].append(picklist_input_id)
                    data['value_token_type_ids'].append(picklist_type_id)
                    data['value_attention_mask'].append(picklist_mask_id)
                    data['value_label'].append(label)

            #  span case
            if "span" in model.mode:
                if value in instance.turn_utter:
                    input_id, token_type_id, attention_mask_id, start, end = preprocess(tokenizer, instance.turn_utter,
                                                                                        slot, value)
                    data['input_ids'].append(input_id)
                    data['token_type_ids'].append(token_type_id)
                    data['attention_mask'].append(attention_mask_id)
                    data['start_positions'].append(start)
                    data['end_positions'].append(end)
                    data['span_mask'].append(1.0)
                    input_id, token_type_id, attention_mask_id, start, end = preprocess(tokenizer, instance.turn_utter,
                                                                                        neg_slot)
                    data['input_ids'].append(input_id)
                    data['token_type_ids'].append(token_type_id)
                    data['attention_mask'].append(attention_mask_id)
                    data['start_positions'].append(start)
                    data['end_positions'].append(end)
                    data['span_mask'].append(0.0)

                else:
                    for sl in [slot, neg_slot]:
                        input_id, token_type_id, attention_mask_id, start, end = preprocess(tokenizer, instance.turn_utter,
                                                                                            sl)
                        data['input_ids'].append(input_id)
                        data['token_type_ids'].append(token_type_id)
                        data['attention_mask'].append(attention_mask_id)
                        data['start_positions'].append(start)
                        data['end_positions'].append(end)
                        data['span_mask'].append(0.0)

        # testing
        # if idx == 20:
        #     break

    # print(torch.tensor(np.array(data["input_ids"])).shape)
    for key in data:
        data[key] = torch.tensor(np.array(data[key]))
    return data


def generate_test_data(instance, tokenizer, ontology, slot_meta, model):
    data = collections.defaultdict(list)
    # for idx, instance in enumerate(train_data_raw):
    # gold_slots = set(["-".join(g.split("-")[:-1]) for g in instance.gold_state])
    input_ids, token_type_ids, attention_masks = [], [], []

    for slot in ontology:
        input_id, token_type_id, attention_mask_id, _, _ = preprocess(tokenizer, instance.turn_utter, slot)
        input_ids.append(input_id)
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask_id)

    gold_slot_value = {'-'.join(ii.split('-')[:-1]): ii.split('-')[-1] for ii in instance.gold_state}
    ops = []
    for j, slot in enumerate(slot_meta):
        if slot not in gold_slot_value:
            ops.append("none")
        else:
            ops.append("update" if gold_slot_value[slot] != "dontcare" else "dontcare")
    #
    # data['input_ids'].append(input_ids)
    # data['token_type_ids'].append(token_type_ids)
    # data['attention_mask'].append(attention_masks)
    # data['gold_state'].append(instance.gold_state)
    # data['gold_op'].append(ops)
    # data['is_last_turn'].append(instance.is_last_turn)

    data['input_ids'] = input_ids
    data['token_type_ids'] = token_type_ids
    data['attention_mask'] = attention_masks
    data['gold_state'] = instance.gold_state
    data['gold_op'] = ops
    data['is_last_turn'] = instance.is_last_turn
    data['turn_id'] = instance.turn_id
    data['id'] = instance.id
    data['turn_domain'] = instance.turn_domain

    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        data[key] = torch.tensor(np.array(data[key]))
    return data


def evaluate_span(model, test_data_raw, tokenizer, ontology, slot_meta, epoch, device):
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
        test_data = generate_test_data(test_data_raw[step], tokenizer, ontology, slot_meta, model)
        if model.mode == "span":
            _inp = {"input_ids": torch.tensor(test_data['input_ids'].to(device)),
                    "attention_mask": torch.tensor(test_data["attention_mask"].to(device)),
                    "token_type_ids": torch.tensor(test_data["token_type_ids"].to(device))}
            #
        elif model.mode == "pickspan":
            _inp = {"input_ids": torch.tensor(test_data['input_ids'].to(device)),
                    "attention_mask": torch.tensor(test_data["attention_mask"].to(device)),
                    "token_type_ids": torch.tensor(test_data["token_type_ids"].to(device)),
                    "value_input_ids": torch.tensor(test_data['value_input_ids'].to(device)),
                    "value_attention_mask": torch.tensor(test_data["value_attention_mask"].to(device)),
                    "value_token_type_ids": torch.tensor(test_data["value_token_type_ids"].to(device))}

        gold_op = test_data['gold_op']
        gold_state = test_data['gold_state']

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**_inp)
        end = time.perf_counter()
        wall_times.append(end - start)

        # slot prediction
        pred_op = [id2op[j] for j in np.argmax(outputs[0].cpu().data.numpy(), 1)]
        # pred_op[0] = 'update'
        #
        # # value prediction
        pred_state = set()
        for j, tmp in enumerate(zip(pred_op, slot_meta)):
            _op, _slot = tmp
            if _op == "none":
                continue
            elif _op == "dontcare":
                pred_state.add(_slot + "-" + _op)
            else:
                # get span prediction to text
                if model.mode == "span":
                    all_tokens = tokenizer.convert_ids_to_tokens(test_data['input_ids'].numpy()[j])
                    start = np.argmax(outputs[1][j].cpu().data.numpy())
                    end = np.argmax(outputs[2][j].cpu().data.numpy())
                    span = ' '.join(all_tokens[start: end + 1])
                    pred_state.add(_slot + "-" + span)
                elif model.mode == "pickspan":
                    pass

        if set(pred_state) == set(gold_state):
            joint_acc += 1
        key = str(test_data['id']) + '_' + str(test_data['turn_id'])

        results[key] = [list(pred_state), gold_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        # Compute operation accuracy
        temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_op, gold_op)]) / len(pred_op)
        op_acc += temp_acc

        if test_data['is_last_turn']:
            final_count += 1
            if set(pred_state) == set(gold_state):
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
