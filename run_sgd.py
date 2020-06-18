import argparse

from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def print2file(path, name, format, printout,enablePrint=True):
    if enablePrint:
        print(printout)
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path + name + format, 'a')
    thefile.write("%s\n" % (printout))
    thefile.close()
if __name__ == '__main__':

    parser = argparse.ArgumentParser('DSTC8 - SGD')
    parser.add_argument('--eval_mode', type=str, default="single")
    parser.add_argument('--enable_wdc', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default="out/")
    parser.add_argument('--use_cuda', type=int, default=0)
    parser.add_argument('--epoch_nb', type=int, default=5)

    args = parser.parse_args()

    # Create the QuestionAnsweringModel
    model = QuestionAnsweringModel('bert', 'bert-base-uncased', use_cuda=args.use_cuda, args={'reprocess_input_data': True, 'overwrite_output_dir': True})
    modelName = "sgd_%s_%s" % (args.eval_mode, "bert")

    if args.enable_wdc:
        model.train_model('data/sgd/wdc.json')
        modelName = "sgd_%s_%s_wdc" % (args.eval_mode, "bert")

    #
    best_acc = 0
    for epoch in range(args.epoch_nb):

        model.train_model('data/sgd/sgd-train-%s.json' % args.eval_mode)
        # result, out = model.eval_model('data/sgd/sgd-dev-%s.json' % args.eval_mode)
        result, out = model.eval_model('data/sgd/sgd-test-%s.json' % args.eval_mode)

        slot_acc = result['correct'] / (result['correct'] + result['similar'] + result['incorrect'])

        if slot_acc > best_acc:
            best_acc = slot_acc

            print2file(args.out_dir, modelName, ".res", slot_acc, True)
            # save output to file
            with open('%s%s.out' % (args.out_dir, modelName), 'w') as f:
                json.dump(out, f)
