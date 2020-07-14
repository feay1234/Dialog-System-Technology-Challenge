import torch
from simpletransformers.question_answering import QuestionAnsweringModel
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel
import collections
import numpy as np
import time
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy


class DST_SPAN():
    def __init__(self, use_cuda=False):
        self.model = QuestionAnsweringModel('bert', 'bert-base-uncased', use_cuda=use_cuda,
                                            args={'reprocess_input_data': True, 'overwrite_output_dir': True})

    def generate_train_data(self, train_data_raw, ontology):
        train_data = []
        for instance in train_data_raw:
            context = instance.dialog_history + instance.turn_utter
            gold_slots = ["-".join(g.split("-")[:-1]) for g in instance.gold_state]
            gold_values = [g.split("-")[-1] for g in instance.gold_state]

            qas = []
            for sid, gold in enumerate(zip(gold_slots, gold_values)):
                slot, value = gold
                did = "%s_t%d_s%d" % (instance.id, instance.turn_id, sid)
                qas.append({'id': did,
                            'is_impossible': False if value in context else True,
                            'question': slot.replace("-", " "),
                            'answers': [
                                {'text': value, 'answer_start': context.index(value) if value in context else 0}]})
                # Negative slot
                neg_slot = np.random.choice(list(ontology.keys()))
                while slot == neg_slot or neg_slot in gold_slots:
                    neg_slot = np.random.choice(list(ontology.keys()))

                qas.append({'id': did + "_neg",
                            'is_impossible': True,
                            'question': neg_slot.replace("-", " "),
                            'answers': [{'text': "", 'answer_start': -1}]})
            train_data.append({"context": context, "qas": qas})
        return train_data





    def evaluate(self, test_data_raw, ontology, slot_meta, epoch):
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
            context = instance.dialog_history + instance.turn_utter

            qas = []
            for idx, slot in enumerate(ontology):
                did = "%s_t%d_s%d" % (instance.id, instance.turn_id, idx)
                qas.append({'id': did,
                            'question': slot.replace("-", " ")})

            test_data = [{"context": context, "qas": qas}]

            gold_slot_value = {'-'.join(ii.split('-')[:-1]): ii.split('-')[-1] for ii in instance.gold_state}
            gold_op = []
            for j, slot in enumerate(slot_meta):
                gold_op.append("none" if slot not in gold_slot_value else "update")


            start = time.perf_counter()
            # prediction
            with torch.no_grad():
                outputs = self.model.predict(test_data)
            end = time.perf_counter()
            wall_times.append(end - start)



            # slot prediction
            pred_op = ['none' if len(pred['answer']) == 0 else "pred" for pred in outputs]

            # # value prediction
            pred_state = set()
            for pred, slot in enumerate(zip(outputs, slot_meta)):
                span = pred['answer']
                if span == "":
                    pred_op.append("none")
                else:
                    pred_op.append("update")
                    pred_state.add(slot + "-" + span)

            gold_state = instance.gold_state

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

            if instance.is_last_turn:
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
