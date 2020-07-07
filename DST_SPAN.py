from transformers import BertPreTrainedModel
import torch
from torch import nn
import numpy as np
from torch.nn import CrossEntropyLoss, BCELoss
from transformers import BertForQuestionAnswering, BertTokenizer, BertModel, AdamW

from transformers import BertPreTrainedModel


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

        value_outputs = self.bert(
            value_input_ids,
            attention_mask=value_attention_mask,
            token_type_ids=value_token_type_ids,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        rCLS = sequence_output[:, 0, :]
        vCLS = value_outputs[0][:, 0, :]

        slot_logits = self.clf(rCLS)
        sigmoid = nn.Sigmoid()
        value_logits = sigmoid(self.binaryCLF(vCLS))

        outputs = (start_logits, end_logits,) + outputs[2:]
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
            value_loss = loss_bce(value_logits, value_label.float())

            total_loss = (start_loss + end_loss + slot_loss + value_loss) / 4
            #             total_loss = start_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


# model = DST_SPAN.from_pretrained('bert-base-uncased')
