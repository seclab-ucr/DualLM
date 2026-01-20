#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from scipy.stats import percentileofscore
from collections import Counter
from multiprocessing import Pool
import sentencepiece as spm
import numpy as np


def main():
    """
    Helper script to encode raw text with the  BPE using multiple processes.
    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.zbpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"
    token_lens = []
    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines,sample_length) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                token_lens.append(sample_length)
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)
        token_lens = np.array(token_lens)
        #print("# of samples: ", len(token_lens))
        percentile= percentileofscore(token_lens, int(args.max_len))
        #print(" {}% of samples are smaller than {} tokens".format(percentile,args.max_len))
        #print(" mean token length: {}".format(np.mean(token_lens)))
        #print(" middle token length: {}".format(np.median(token_lens)))
        #print(" 0%% token length: {}".format(np.percentile(token_lens,0)))
        #print(" 25%% token length: {}".format(np.percentile(token_lens,25)))
        #print(" 50%% token length: {}".format(np.percentile(token_lens,50)))
        #print(" 75%% token length: {}".format(np.percentile(token_lens,75)))
        #print(" 100%% token length: {}".format(np.percentile(token_lens,100)))

class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global sp
        sp = spm.SentencePieceProcessor(model_file=self.args.model_file)
       

    def encode(self, line):
        global sp
        return sp.encode(line, out_type=str)

    def decode(self, tokens):
        global sp
        return sp.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        sample_length = 0
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None,sample_length]
            tokens = self.encode(line)
            sample_length = len(tokens)
            
            tokens = tokens[:self.args.max_len]#truncate the data sample; we need to design another way for it since our data sample format is different
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines,sample_length]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

if __name__ == "__main__":
    main()