from pytorch_transformers import *
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss

BertForQuestionAnswering
class DST_SPAN(BertPreTrainedModel):

    def __init__(self, config):
        super(DST_SPAN, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.clf = nn.Linear(config.hidden_size, 3)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, position_ids=None, head_mask=None, slot_positions=None, slot_value_targets=None,
                slot_value_candidates=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        r_cls = sequence_output[:, 0, :]
        slot_logits = self.clf(r_cls)

        # nn.max(0, CosineSimilarity)

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
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        slot_loss = loss_fct(slot_logits, slot_positions)
        total_loss = (slot_loss + start_loss + end_loss) / 3
        outputs = (total_loss,) + outputs

        return total_loss, start_logits, end_logits, sequence_output  # (loss), start_logits, end_logits, (hidden_states), (attentions)


model = DST_SPAN.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer(" ".join(["the"]*700), return_tensors="pt")
