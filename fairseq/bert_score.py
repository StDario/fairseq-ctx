# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import ctypes
import math
import torch

# import logging
# logging.getLogger("transformers").setLevel(logging.WARNING)


# try:
#     from bert_score import score, score_custom
# except ImportError as e:
#     import sys
#     sys.stderr.write('ERROR: missing bert_score. run `pip install bert_score .`\n')
#     raise e


class SacrebleuScorer(object):
    def __init__(self):
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.sys.append(pred)

    def score(self, order=4):
        return self.result_string(order).score

    def result_string(self, order=4):
        if order != 4:
            raise NotImplementedError
        return self.sacrebleu.corpus_bleu(self.sys, [self.ref])


class Scorer(object):

    def __init__(self, pad, eos, unk, model=None, model_type='xlm-mlm-100-1280', trg_lang=None, model_device=0):

        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.refs = []
        self.hyps = []
        self.model_type = model_type
        self.trg_lang = trg_lang
        self.model_device = model_device
        self.bert_score_model = model
        self.tokenizer = None
        self.stats_dict = dict()
        self.reset()

    def reset(self):
        self.refs = []
        self.hyps = []

    def add(self, ref, pred):
        self.refs.append(ref)
        self.hyps.append(pred)

    def score(self):
        costs = score_custom(self.refs, self.hyps, model_type=self.model_type, lang=self.trg_lang, verbose=False,
                             model=self.bert_score_model, device=torch.device("cuda", self.model_device),
                             tokenizer=self.tokenizer)
                             # stats_dict=self.stats_dict)

        if self.tokenizer is None:
            self.tokenizer = costs[4]

        # if self.stats_dict is None:
        #     self.stats_dict = costs[5]

        # self.bert_score_model = costs[3] if self.bert_score_model is None else self.bert_score_model

        return costs[2] # .sum()
        # psum = sum(math.log(p) if p > 0 else float('-Inf')
        #            for p in self.precision()[:order])
        # return self.brevity() * math.exp(psum / order) * 100

    def result_string(self):
        return "Empty string"
        # assert order <= 4, "BLEU scores for order > 4 aren't supported"
        # fmt = 'BLEU{} = {:2.2f}, {:2.1f}'
        # for _ in range(1, order):
        #     fmt += '/{:2.1f}'
        # fmt += ' (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})'
        # bleup = [p * 100 for p in self.precision()[:order]]
        # return fmt.format(order, self.score(order=order), *bleup,
        #                   self.brevity(), self.stat.predlen/self.stat.reflen,
        #                   self.stat.predlen, self.stat.reflen)
