from tqdm import tqdm
from time import strftime, localtime

from DST_SPAN import DST_SPAN, generate_train_data, evaluate_span
from model import SomDST
from transformers import BertTokenizer, BertModel, AdamW

from utils.ckpt_utils import download_ckpt
from utils.data_utils import prepare_dataset, load_data, save_result_to_file
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from evaluation import model_evaluation
import torch
import numpy as np
import argparse
import random
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    ontology = json.load(open(args.ontology_data))
    slot_meta, ontology = make_slot_meta(ontology)
    op2id = OP_SET[args.op_code]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if os.path.exists(args.train_data_path + ".pk"):
        train_data_raw = load_data(args.train_data_path + ".pk")
    else:

        train_data_raw = prepare_dataset(data_path=args.train_data_path,
                                         tokenizer=tokenizer,
                                         slot_meta=slot_meta,
                                         n_history=args.n_history,
                                         max_seq_length=args.max_seq_length,
                                         op_code=args.op_code)

    print("# train examples %d" % len(train_data_raw))
    # maxlen = 0
    # for i in range(len(train_data_raw)):
    #     maxlen = max(maxlen, len(train_data_raw[i].turn_utter.split()))
    # print(maxlen)

    if os.path.exists(args.dev_data_path + ".pk"):
        dev_data_raw = load_data(args.dev_data_path + ".pk")
    else:
        dev_data_raw = prepare_dataset(data_path=args.dev_data_path,
                                       tokenizer=tokenizer,
                                       slot_meta=slot_meta,
                                       n_history=args.n_history,
                                       max_seq_length=args.max_seq_length,
                                       op_code=args.op_code)
    print("# dev examples %d" % len(dev_data_raw))

    if os.path.exists(args.test_data_path + ".pk"):
        test_data_raw = load_data(args.test_data_path + ".pk")
    else:
        test_data_raw = prepare_dataset(data_path=args.test_data_path,
                                        tokenizer=tokenizer,
                                        slot_meta=slot_meta,
                                        n_history=args.n_history,
                                        max_seq_length=args.max_seq_length,
                                        op_code=args.op_code)
    print("# test examples %d" % len(test_data_raw))

    model = DST_SPAN.from_pretrained('bert-base-uncased')

    if not os.path.exists(args.bert_ckpt_path):
        args.bert_ckpt_path = download_ckpt(args.bert_ckpt_path, args.bert_config_path, 'assets')

    ckpt = torch.load(args.bert_ckpt_path, map_location='cpu')
    model.bert.load_state_dict(ckpt)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    model.to(device)

    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)

    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    best_score = {'epoch': float("-inf"), 'joint_acc': float("-inf"), 'op_acc': float("-inf"),
                  'final_slot_f1': float("-inf")}

    best_epoch = 0
    for epoch in range(args.n_epochs):
        batch_loss = []
        model.train()
        # for step in tqdm(range(int(len(train_data_raw) / args.batch_size) + 1), desc="training"):
        #     train_data = generate_train_data(train_data_raw[step * args.batch_size:(step * args.batch_size) + args.batch_size], ontology, tokenizer)
        for step in tqdm(range(len(train_data_raw)), desc="training"):
            train_data = generate_train_data(train_data_raw[step: step+1], ontology, tokenizer)

            # ignore dialogue with no trainable turns
            if len(train_data['input_ids']) == 0:
                continue

            _inp = {"input_ids": train_data['input_ids'].to(device),
                    "attention_mask": train_data['attention_mask'].to(device),
                    "token_type_ids": train_data['token_type_ids'].to(device),
                    "start_positions": train_data['start_positions'].to(device),
                    "end_positions": train_data['end_positions'].to(device),
                    "span_mask": train_data['span_mask'].to(device),
                    "slot_label": train_data['slot_label'].to(device)}

            outputs = model(**_inp)

            loss = outputs[0].mean()
            batch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            model.zero_grad()
        #
            if step % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" % (epoch + 1, args.n_epochs, step, len(train_data_raw), np.mean(batch_loss)))
                batch_loss = []
    #
        if (epoch + 1) % args.eval_epoch == 0:
            eval_res, res_per_domain, pred  = evaluate_span(model, dev_data_raw, tokenizer, ontology, slot_meta, epoch+1)
    #
            if eval_res['joint_acc'] > best_score['joint_acc']:
                best_score = eval_res
                model_to_save = model.module if hasattr(model, 'module') else model
                save_path = os.path.join(args.out_dir, args.filename + '.bin')
                torch.save(model_to_save.state_dict(), save_path)
                best_epoch = epoch + 1
            print("Best Score : ", best_score)
            print("\n")

    print("Test using best model...")
    ckpt_path = os.path.join(args.out_dir, args.filename + '.bin')
    model = DST_SPAN.from_pretrained('bert-base-uncased')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    eval_res, res_per_domain, pred = evaluate_span(model, dev_data_raw, tokenizer, ontology, slot_meta, best_epoch )
    # save to file
    save_result_to_file(args.out_dir + "/" + args.filename + ".res", eval_res, res_per_domain)
    json.dump(pred, open('%s.pred' % (args.out_dir + "/" + args.filename), 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_root", default='data/multiwoz2.1', type=str)
    parser.add_argument("--train_data", default='train_dials.json', type=str)
    # parser.add_argument("--train_data", default='wdc_dials.json', type=str)
    parser.add_argument("--dev_data", default='dev_dials.json', type=str)
    parser.add_argument("--test_data", default='test_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='assets/bert-base-uncased-pytorch_model.bin', type=str)
    parser.add_argument("--save_dir", default='outputs', type=str)
    parser.add_argument("--out_dir", default='outputs', type=str)
    parser.add_argument('--enable_wdc', type=int, default=0)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    # parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--not_shuffle_state", default=False, action='store_true')
    parser.add_argument("--shuffle_p", default=0.5, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--msg", default=None, type=str)
    parser.add_argument("--exclude_domain", default=False, action='store_true')

    args = parser.parse_args()

    dataset = args.data_root.split("/")[1]
    modelName = "span"
    timestamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())

    filename = "%s_%s_e%d_%s" % (dataset, modelName, args.n_epochs, timestamp)

    if args.enable_wdc:
        filename = "%s_%s_wdc_e%d_%s" % (dataset, modelName, args.n_epochs, timestamp)

    args.save_dir = os.path.join(args.save_dir, filename)
    args.filename = filename

    args.train_data_path = os.path.join(args.data_root, args.train_data)
    args.dev_data_path = os.path.join(args.data_root, args.dev_data)
    args.test_data_path = os.path.join(args.data_root, args.test_data)
    args.ontology_data = os.path.join(args.data_root, args.ontology_data)
    args.shuffle_state = False if args.not_shuffle_state else True
    print('pytorch version: ', torch.__version__)
    print(args)
    main(args)
