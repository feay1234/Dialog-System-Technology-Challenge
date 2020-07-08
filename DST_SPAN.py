import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel
import collections
import numpy as np

class DST_SPAN(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.clf = nn.Linear(config.hidden_size, 3)
        self.binaryCLF = nn.Linear(config.hidden_size, 1)

        self.init_weights()

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





def preprocess(tokenizer, context, slot=None, value=None):
    max_len = 512
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


def generate_train_data(train_data_raw, ontology, tokenizer):
    data = collections.defaultdict(list)
    # pbar = tqdm(enumerate(train_data_raw), total=len(train_data_raw), desc="Generating training dataset", ncols=0)

    for idx, instance in enumerate(train_data_raw):
    # for idx, instance in pbar:

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

            #             #   picklist case
            #             cand_values = ontology[slot]
            #             neg_value = np.random.choice(cand_values)
            #             while neg_value == value or neg_value == "do n't care":
            #                 neg_value = np.random.choice(cand_values)

            #             for val, label in [(value,1.0), (neg_value, 0.0)]:
            #                 picklist_input_id, picklist_type_id, picklist_mask_id, _, _  = preprocess(val)
            #                 data['value_input_ids'].append(picklist_input_id)
            #                 data['value_token_type_ids'].append(picklist_type_id)
            #                 data['value_attention_mask'].append(picklist_mask_id)
            #                 data['value_label'].append(label)

            # #           span case
            span_mask = []
            if value in instance.turn_utter:
                input_id, token_type_id, attention_mask_id, start, end = preprocess(tokenizer, instance.turn_utter, slot, value)
                data['input_ids'].append(input_id)
                data['token_type_ids'].append(token_type_id)
                data['attention_mask'].append(attention_mask_id)
                data['start_positions'].append(start)
                data['end_positions'].append(end)
                data['span_mask'].append(1.0)
                input_id, token_type_id, attention_mask_id, start, end = preprocess(tokenizer, instance.turn_utter, neg_slot)
                data['input_ids'].append(input_id)
                data['token_type_ids'].append(token_type_id)
                data['attention_mask'].append(attention_mask_id)
                data['start_positions'].append(start)
                data['end_positions'].append(end)
                data['span_mask'].append(0.0)

            else:
                for sl in [slot, neg_slot]:
                    input_id, token_type_id, attention_mask_id, start, end = preprocess(tokenizer, instance.turn_utter, sl)
                    data['input_ids'].append(input_id)
                    data['token_type_ids'].append(token_type_id)
                    data['attention_mask'].append(attention_mask_id)
                    data['start_positions'].append(start)
                    data['end_positions'].append(end)
                    data['span_mask'].append(0.0)

        # testing
        # if idx == 20:
        #     break

    for key in data:
        data[key] = torch.tensor(np.array(data[key]))
    return data

