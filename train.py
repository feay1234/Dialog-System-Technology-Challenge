from tqdm import tqdm
from time import strftime, localtime

from transformers import BertForQuestionAnswering

from model import SomDST
from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule, BertConfig
from utils.data_utils import prepare_dataset, MultiWozDataset, load_data, save_result_to_file
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from utils.ckpt_utils import download_ckpt, convert_ckpt_compatible
from evaluation import model_evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
import os
import json
import time

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
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    print(op2id)

    if os.path.exists(args.train_data_path + ".pk"):
        train_data_raw = load_data(args.train_data_path + ".pk")
    else:

        train_data_raw = prepare_dataset(data_path=args.train_data_path,
                                         tokenizer=tokenizer,
                                         slot_meta=slot_meta,
                                         n_history=args.n_history,
                                         max_seq_length=args.max_seq_length,
                                         op_code=args.op_code)
    train_data = MultiWozDataset(train_data_raw,
                                 tokenizer,
                                 slot_meta,
                                 args.max_seq_length,
                                 rng,
                                 ontology,
                                 args.word_dropout,
                                 args.shuffle_state,
                                 args.shuffle_p)
    print("# train examples %d" % len(train_data_raw))
    print(len(train_data))

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

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob

    model = SomDST(model_config, len(op2id), len(domain2id), op2id['update'], args.exclude_domain)

    if not os.path.exists(args.bert_ckpt_path):
        args.bert_ckpt_path = download_ckpt(args.bert_ckpt_path, args.bert_config_path, 'assets')

    ckpt = torch.load(args.bert_ckpt_path, map_location='cpu')
    model.encoder.bert.load_state_dict(ckpt)

    # re-initialize added special tokens ([SLOT], [NULL], [EOS])
    model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
    model.to(device)

    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.encoder.named_parameters())
    enc_optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr)
    enc_scheduler = WarmupLinearSchedule(enc_optimizer, int(num_train_steps * args.enc_warmup),
                                         t_total=num_train_steps)

    dec_param_optimizer = list(model.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
    dec_scheduler = WarmupLinearSchedule(dec_optimizer, int(num_train_steps * args.dec_warmup),
                                         t_total=num_train_steps)

    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    loss_fnc = nn.CrossEntropyLoss()
    best_score = {'epoch': float("-inf"), 'joint_acc_score': float("-inf"), 'op_acc': float("-inf"),
                  'final_slot_f1': float("-inf")}

    for epoch in range(args.n_epochs):
        batch_loss = []
        model.train()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="training", ncols=0)
        for step, batch in pbar:
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, input_mask, segment_ids, state_position_ids, op_ids, \
            domain_ids, gen_ids, max_value, max_update = batch

            if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
                teacher = gen_ids
            else:
                teacher = None

            domain_scores, state_scores, gen_scores = model(input_ids=input_ids,
                                                            token_type_ids=segment_ids,
                                                            state_positions=state_position_ids,
                                                            attention_mask=input_mask,
                                                            max_value=max_value,
                                                            op_ids=op_ids,
                                                            max_update=max_update,
                                                            teacher=teacher)


            loss_s = loss_fnc(state_scores.view(-1, len(op2id)), op_ids.view(-1))
            loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                    gen_ids.contiguous(),
                                                    tokenizer.vocab['[PAD]'])

            loss = loss_s + loss_g
            if args.exclude_domain is not True:
                loss_d = loss_fnc(domain_scores.view(-1, len(domain2id)), domain_ids.view(-1))
                loss = loss + loss_d
            batch_loss.append(loss.item())

            loss.backward()
            enc_optimizer.step()
            enc_scheduler.step()
            dec_optimizer.step()
            dec_scheduler.step()
            model.zero_grad()

            if step % 100 == 0:
                if args.exclude_domain is not True:
                    print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f, dom_loss : %.3f" \
                          % (epoch + 1, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss),
                             loss_s.item(), loss_g.item(), loss_d.item()))
                else:
                    print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f" \
                          % (epoch + 1, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss),
                             loss_s.item(), loss_g.item()))
                batch_loss = []

        if (epoch + 1) % args.eval_epoch == 0:
            eval_res, res_per_domain, pred = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, epoch + 1, args.op_code)

            if eval_res['joint_acc_score'] > best_score['joint_acc_score']:
                best_score['joint_acc_score'] = eval_res['joint_acc_score']
                model_to_save = model.module if hasattr(model, 'module') else model
                save_path = os.path.join(args.out_dir, args.filename + '.bin')
                torch.save(model_to_save.state_dict(), save_path)
            print("Best Score : ", best_score)
            print("\n")

    print("Test using best model...")
    best_epoch = best_score['epoch']
    ckpt_path = os.path.join(args.out_dir, args.filename + '.bin')
    model = SomDST(model_config, len(op2id), len(domain2id), op2id['update'], args.exclude_domain)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    best_epoch = 0 #dummpy
    eval_res, res_per_domain, pred = model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code)
    # save to file
    save_result_to_file(args.out_dir + "/" + args.filename + ".res", eval_res, res_per_domain)
    json.dump(pred, open('%s.pred' % (args.out_dir + "/" + args.filename), 'w'))

    # model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
    #                  is_gt_op=False, is_gt_p_state=False, is_gt_gen=True)
    # model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
    #                  is_gt_op=False, is_gt_p_state=True, is_gt_gen=False)
    # model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
    #                  is_gt_op=False, is_gt_p_state=True, is_gt_gen=True)
    # model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
    #                  is_gt_op=True, is_gt_p_state=False, is_gt_gen=False)
    # model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
    #                  is_gt_op=True, is_gt_p_state=True, is_gt_gen=False)
    # model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
    #                  is_gt_op=True, is_gt_p_state=False, is_gt_gen=True)
    # model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
    #                  is_gt_op=True, is_gt_p_state=True, is_gt_gen=True)


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
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=1, type=int)
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
    modelName = "som"
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
