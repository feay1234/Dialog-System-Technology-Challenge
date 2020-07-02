from transformers import BertTokenizer, BertForQuestionAnswering
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute [SEP] don't know", return_tensors="pt")
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
loss, start_scores, end_scores = outputs[:3]
print(inputs)