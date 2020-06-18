from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('bert', 'bert-base-uncased', use_cuda=False, args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'use_cuda': True})

model.train_model('data/wdc.json')
