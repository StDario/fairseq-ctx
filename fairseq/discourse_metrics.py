# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import ctypes
import math
import torch


class DiscourseScorer(object):

    def __init__(self):
        self.reset()
        self.coherence_scorer = CoherenceScorer()
        self.coreference_scorer = CoreferenceScorer()

    def reset(self):
        self.ref = []
        self.sys = []
        self.src = []

    def add(self, ref, pred, src):
        self.ref.append(ref)
        self.sys.append(pred)
        self.src.append(src)

    def score(self, coherence=True, coreference=True):

        score = 0
        if coherence:
            score += self.coherence_scorer.score(self.ref, self.sys, self.src)
        if coreference:
            score += self.coreference_scorer.score(self.ref, self.sys)

        return score


class CoreferenceScorer(object):

    def __init__(self, pronouns):
        self.pronouns = pronouns

    def score(self, ref, pred, src):

        ref_pronouns = []

        for sent_ref in ref:
            ref_pronouns.append([])
            for i, w_ref in enumerate(sent_ref):
                if w_ref in self.pronouns:
                    ref_pronouns[-1].append((w_ref, i))

        scores = {}

        for j, sent_pred in enumerate(pred):
            for i, w_pred in enumerate(sent_pred):
                if w_pred in self.pronouns:

                    if w_pred not in scores:
                        scores[w_pred] = [0, 0]

                    ref_tokens = [rp[0] for rp in ref_pronouns[j] if abs(rp[1] - i) < 5]
                    correct_tokens = [rp[0] for rp in ref_pronouns[j] if rp[0] == w_pred and abs(rp[1] - i) < 5]

                    scores[w_pred][0] += len(correct_tokens)
                    scores[w_pred][1] += len(ref_tokens)

        correct = 0
        all = 0

        for token in scores:
            correct += token[0]
            all += token[1]

        return correct * 1. / all


class CoherenceScorer(object):

    def __init__(self):
        pass

    def score(self, ref, pred, src):



        return 0