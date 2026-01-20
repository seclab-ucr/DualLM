import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from fairseq.data import data_utils
from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
from pathlib import Path
import sys
sys.path.append("../../")
import helper
import ast
import os
import time

cwd = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = str(Path(cwd).parent.parent.absolute())
linux_dir = PROJECT_DIR + "/linuxs/linux"
HOME_DIR = str(Path(cwd).parent.absolute())


# https://github.com/pytorch/fairseq/blob/108f7204f6ccddb676e6d52006da219ce96a02dc/fairseq/models/bart/hub_interface.py#L33
def encode(model, sentence, max_positions=512):
    tokens = sentence
    if len(tokens.split(" ")) > max_positions - 2:
        tokens = " ".join(tokens.split(" ")[: max_positions - 2])
    bpe_sentence = "<s> " + tokens + " </s>"
    tokens = model.task.source_dictionary.encode_line(
        bpe_sentence, add_if_not_exist=False, append_eos=False
    )
    return tokens.long()




if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--recalculate', action='store_true')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--data_bin_path', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output', default='result.txt')
    parser.add_argument('--label_file', required=True)
    parser.add_argument('--classification_head_name', default='sentence_classification_head')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_example', default=-1, type=int)
    parser.add_argument('--num_classes', default=6, type=int)
    args = parser.parse_args()
   
    bart = RobertaModel.from_pretrained(
        args.model_dir,
        checkpoint_file=args.model_name,
        data_name_or_path=args.data_bin_path,
        task="sentence_prediction",
    )
    # print(bart.task.label_dictionary.nspecial)
    label_fn = lambda label: bart.task.label_dictionary.string(
        [label + bart.task.label_dictionary.nspecial]
    )
    global is_second
    is_second = False
    if "second" in args.model_name:
        is_second = True
    bart.cuda()
    bart.eval()
    ncorrect, nsamples = 0.0, 0.0
    kcorrect = 0.0
    y_true, y_pred = [], []
    commit_file = args.label_file.replace(".label","_commits")
 
    failed_commits = []
    top_k = 2
    report = str(args.model_dir)+"/"+str(args.model_name) + "\n"
    report += "Predicted Label, True Label, Predicted Label Name, True Label Name, Commit\n"
    start_time = time.time()
    with open(args.input_file) as inpf, open(args.label_file) as labelf, open(args.output, 'w') as outp, open(commit_file) as commitf:
        inputs = inpf.readlines()
        labels = labelf.readlines()
        commits = commitf.readlines()
        assert len(inputs) == len(commits) , "input and commit file should have the same number of lines"
        if args.max_example != -1 and args.max_example > 0:
            inputs = inputs[:args.max_example]
            labels = labels[:args.max_example]
            commits = commits[:args.max_example]
 
        total_batches = int(math.ceil(len(inputs) / args.batch_size))
        start = 0
        start_indices = np.arange(0, len(inputs), args.batch_size)
        log_res = ""
        for start in tqdm(start_indices, total=total_batches):
            batch_input = []
            batch_targets = []
            eval_commits = []
            batch_size = args.batch_size if start + args.batch_size <= len(inputs) \
                else len(inputs) - start
            for idx in range(start, start + batch_size):
                line = inputs[idx].strip()
                if "," in labels[idx]:
                    label = int(labels[idx].split(",")[0])
                else:
                    label = int(labels[idx])
                tokens = encode(bart, line)
                batch_input.append(tokens)
                batch_targets.append(label)
                eval_commits.append(commits[idx].strip())

            y_true.extend(batch_targets)
            with torch.no_grad():
                batch_input = data_utils.collate_tokens(
                    batch_input, bart.model.encoder.dictionary.pad(), left_pad=False
                )
                prediction = bart.predict(args.classification_head_name, batch_input)
                prediction1 = prediction
                for p, t in zip(torch.topk(prediction,top_k)[1], batch_targets):
                    for p1 in p:
                        if int(label_fn(p1))  == t:
                            kcorrect += 1
                            break
                # print(kcorrect)
                
                prediction = prediction.argmax(dim=1).cpu().numpy().tolist()
                prediction = [int(label_fn(p)) for p in prediction]
                y_pred.extend(prediction)
                # print(len(prediction))
                ncorrect += sum([int(p == t) for p, t in zip(prediction, batch_targets)])
                assert len(prediction) == len(batch_targets), "prediction and target length not match"
                for index in range(len(prediction)):
                    p = prediction[index]
                    t = batch_targets[index]
                    if p != t:
                        # report += str(p) + " " + str(t) + " " + str(target_names[p]) + " " + str(target_names[t]) + " " + str(eval_commits[index]) + " "+str(prediction1[index].cpu().numpy().tolist()) + " "+str(prediction1[index].argmax())+" "+str(label_fn(prediction1[index].argmax()))+"\n"
                        failed_commits.append(eval_commits[index])
                        
                # print(ncorrect)
                nsamples += len(prediction)
                log = ['{}\t{}\t{}\t{}'.format(p, t, commit,str(prediction1_a.cpu().numpy().tolist()) ) for p, t , commit, prediction1_a in zip(prediction, batch_targets, eval_commits,prediction1)]
                log_res += '\n'.join(log) + '\n'
                
        
        assert len(inputs) == nsamples

        acc = round(100.0 * ncorrect / nsamples, 2)

        acc = round(100.0 * kcorrect / nsamples, 2)
        
        # report += classification_report(y_true, y_pred, target_names=target_names, digits=3)
        # print(report)
        outp.write(report + '\n')
        outp.write(log_res)
       
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"The function took {elapsed_time} seconds to complete.")