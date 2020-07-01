import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel, BertConfig

from keras import backend as K

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


max_len = 512
configuration = BertConfig()  # default paramters and configuration for BERT

"""
## Set-up BERT tokenizer
"""
# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

def preprocess(context, question):
    context = context
    question = question

    # Clean context, answer and question
    context = " ".join(str(context).split())
    question = " ".join(str(question).split())


    # Tokenize context

    tokenized_context = tokenizer.encode(context)

    # Tokenize question
    question = question
    tokenized_question = tokenizer.encode(question)

    # Create inputs
    input_ids = tokenized_context.ids + tokenized_question.ids[1:]
    token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
        tokenized_question.ids[1:]
    )
    attention_mask = [1] * len(input_ids)

    # Pad and create attention masks.
    # Skip if truncation is needed
    padding_length = max_len - len(input_ids)
    if padding_length > 0:  # pad
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

    return input_ids, token_type_ids, attention_mask


def create_model():
    ## BERT encoder
    encoder = TFBertModel.from_pretrained("bert-base-uncased")

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    gate_logits = layers.Dense(3, name="gate_prediction")(embedding[:, 0, :])
    gate_probs = layers.Activation(keras.activations.softmax)(gate_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[gate_probs],
    )
    CCloss = keras.losses.CategoricalCrossentropy()

    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=CCloss)
    return model



with open("data/multiwoz2.1/ontology.json") as json_file:
    ontology = json.load(json_file)

train = json.load(open("data/multiwoz2.1/train_dials.json"))

def generate_slot_gate_clf_woz(data):
    train_data = {}
    for i in range(len(data)):
        dialogue_id = data[i]['dialogue_idx']
        all_context = []
        instances = []
        for t in range(len(data[i]['dialogue'])):
            dialog = data[i]['dialogue'][t]
            belief_state = dialog['belief_state']

            service = dialog['domain']
            if dialog['system_transcript'] != "":
                all_context.append("System: " + dialog['system_transcript'])
            all_context.append("User: " + dialog['transcript'])
            current_slot_names = [i['slots'][0][0] for i in belief_state]
            instances.append([" ".join(all_context), current_slot_names])

        train_data[dialogue_id] = instances
        # break
    return train_data

raw_data = generate_slot_gate_clf_woz(train)


# generate slot classification
x_train, y_train = [[],[],[]], []
pbar = tqdm(raw_data, total=100, desc="training", ncols=0)
# for step, batch in pbar:
count = 0
for did in pbar:
    for turn in raw_data[did]:
        for slot in ontology:
            context = turn[0]
            question = turn[1]
            input_ids, token_type_ids, attention_mask = preprocess(context, question)

            x_train[0].append(input_ids)
            x_train[1].append(token_type_ids)
            x_train[2].append(attention_mask)
            y_train.append([1,0,0] if slot in question else [0,0,1])
    count += 1
    if count == 100:
        break

x_train[0] = np.array(x_train[0])
x_train[1] = np.array(x_train[1])
x_train[2] = np.array(x_train[2])
y_train = np.array(y_train)


print(y_train.shape)
model = create_model()
# context = "User: am looking for a place to to stay that has cheap price range it should be in a type of hotel"
# question = "hotel-price range"
#
# input_ids, token_type_ids, attention_mask = preprocess(context, question)
#
# print(model.predict([np.array([input_ids]), np.array([token_type_ids]), np.array([attention_mask])]))
model.fit(x_train, y_train, batch_size=4)

