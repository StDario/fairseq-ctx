# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
from copy import deepcopy

import numpy as np
import torch
# from mosestokenizer import MosesDetokenizer

from fairseq import bleu, utils, bert_score
import fairseq.discourse_metrics as dm
from fairseq.data import Dictionary, language_pair_dataset
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks import register_task, translation


class BleuScorer(object):

    key = 'bleu'

    def __init__(self, tgt_dict, bpe_symbol='@@ '):
        self.tgt_dict = tgt_dict
        self.bpe_symbol = bpe_symbol
        self.scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        # use a fresh Dictionary for scoring, so that we can add new elements
        self.scoring_dict = Dictionary()

    def preprocess_ref(self, ref):
        ref = self.tgt_dict.string(ref, bpe_symbol=self.bpe_symbol, escape_unk=True)
        return self.scoring_dict.encode_line(ref, add_if_not_exist=True)

    def preprocess_hypo(self, hypo):
        hypo = hypo['tokens']
        hypo = self.tgt_dict.string(hypo.int().cpu(), bpe_symbol=self.bpe_symbol)
        return self.scoring_dict.encode_line(hypo, add_if_not_exist=True)

    def get_cost(self, ref, hypo):
        self.scorer.reset(one_init=True)
        self.scorer.add(ref, hypo)
        return 1. - (self.scorer.score() / 100.)

    def get_reward(self, ref, hypo):
        self.scorer.reset(one_init=True)
        self.scorer.add(ref, hypo)
        return self.scorer.score() / 100.

    def postprocess_costs(self, costs):
        return costs

class DiscourseScorer(object):

    key = 'discourse'

    def __init__(self, tgt_dict, bpe_symbol='@@ '):
        self.tgt_dict = tgt_dict
        self.bpe_symbol = bpe_symbol
        self.scorer = dm.DiscourseScorer()

    def preprocess_ref(self, ref):
        ref = self.tgt_dict.string(ref, bpe_symbol=self.bpe_symbol, escape_unk=True)

        return ref

    def preprocess_hypo(self, hypo):
        hypo = hypo['tokens']
        hypo = self.tgt_dict.string(hypo.int().cpu(), bpe_symbol=self.bpe_symbol)
        return hypo

    def add_ref_hyp(self, ref, hypo):
        self.scorer.add(ref, hypo)

    def get_cost(self):
        res = 1. - self.scorer.score()
        self.scorer.reset()
        return res

    def postprocess_costs(self, costs):
        return costs


class BertScoreScorer(object):

    key = 'bert_score'

    def __init__(self, tgt_dict, bpe_symbol='@@ ', model=None, model_type='xlm-mlm-100-1280', trg_lang=None, model_device=0):

        self.tgt_dict = tgt_dict
        self.bpe_symbol = bpe_symbol
        self.scorer = bert_score.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk(), model=model, model_type=model_type,
                                        trg_lang=trg_lang,
                                        model_device=model_device)
        self.scoring_dict = Dictionary()
        self.detokenizer = MosesDetokenizer(trg_lang)

    def preprocess_ref(self, ref):
        ref = self.tgt_dict.string(ref, bpe_symbol=self.bpe_symbol, escape_unk=True)
        ref = self.detokenizer(ref.split())
        return ref

    def preprocess_hypo(self, hypo):
        hypo = hypo['tokens']
        hypo = self.tgt_dict.string(hypo.int().cpu(), bpe_symbol=self.bpe_symbol)
        hypo = self.detokenizer(hypo.split())
        return hypo

    def add_ref_hyp(self, ref, hypo):
        self.scorer.add(ref, hypo)

    def get_cost(self):
        res = 1. - self.scorer.score()
        self.scorer.reset()
        return res

    def postprocess_costs(self, costs):
        return costs



@register_task('translation_struct')
class TranslationStructuredPredictionTask(translation.TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Compared to :class:`TranslationTask`, this version performs
    generation during training and computes sequence-level losses.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        translation.TranslationTask.add_args(parser)
        parser.add_argument('--seq-beam', default=5, type=int, metavar='N',
                            help='beam size for sequence training')
        parser.add_argument('--seq-keep-reference', default=False, action='store_true',
                            help='retain the reference in the list of hypos')
        parser.add_argument('--seq-scorer', default='bleu', metavar='SCORER',
                            choices=['bleu', 'bert_score'],
                            help='optimization metric for sequence level training')

        parser.add_argument('--bert-score-model-type', default='xlm-mlm-100-1280', choices=['xlm-mlm-100-1280'],
                            help='BERT model to use for BERTScore')
        parser.add_argument('--bert-score-model-device-id', default=0, type=int,
                            help='GPU device ID to put BERTScore model on')

        parser.add_argument('--seq-gen-with-dropout', default=False, action='store_true',
                            help='use dropout to generate hypos')
        parser.add_argument('--seq-max-len-a', default=0, type=float, metavar='N',
                            help='generate sequences of maximum length ax + b, '
                                 'where x is the source length')
        parser.add_argument('--seq-max-len-b', default=200, type=int, metavar='N',
                            help='generate sequences of maximum length ax + b, '
                                 'where x is the source length')
        parser.add_argument('--seq-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE tokens before scoring')
        parser.add_argument('--seq-sampling', default=False, action='store_true',
                            help='use sampling instead of beam search')
        parser.add_argument('--seq-unkpen', default=0, type=float,
                            help='unknown word penalty to be used in seq generation')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self._generator = None
        self._scorers = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return super(TranslationStructuredPredictionTask, cls).setup_task(args, **kwargs)

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions
        criterion = criterions.build_criterion(args, self)
        assert isinstance(criterion, criterions.FairseqSequenceCriterion)
        return criterion

    def build_token_level_criterion(self, args):

        from fairseq import criterions
        original_criterion = args.criterion
        args.criterion = args.token_level_criterion
        criterion = criterions.build_criterion(args, self)
        assert isinstance(criterion, criterions.FairseqCriterion)
        args.criterion = original_criterion
        return criterion

    def build_sequence_level_criterion(self, args):

        from fairseq import criterions
        original_criterion = args.criterion
        args.criterion = args.sequence_level_criterion
        criterion = criterions.build_criterion(args, self)
        assert isinstance(criterion, criterions.FairseqCriterion)
        args.criterion = original_criterion
        return criterion

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        # control dropout during generation
        model.train(self.args.seq_gen_with_dropout)

        # generate hypotheses
        self._generate_hypotheses(model, sample)

        return super().train_step(
            sample=sample,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            ignore_grad=ignore_grad,
        )

    def valid_step(self, sample, model, criterion):
        model.eval()
        self._generate_hypotheses(model, sample)
        return super().valid_step(sample=sample, model=model, criterion=criterion)

    def _generate_hypotheses(self, model, sample):
        # initialize generator
        if self._generator is None:
            self._generator = SequenceGenerator(
                self.target_dictionary,
                beam_size=self.args.seq_beam,
                max_len_a=self.args.seq_max_len_a,
                max_len_b=self.args.seq_max_len_b,
                unk_penalty=self.args.seq_unkpen,
                sampling=self.args.seq_sampling,
            )

        # generate hypotheses
        sample['hypos'] = self._generator.generate(
            [model],
            sample,
        )

        # add reference to the set of hypotheses
        if self.args.seq_keep_reference:
            self.add_reference_to_hypotheses(sample)

    def add_reference_to_hypotheses_(self, sample):
        """
        Add the reference translation to the set of hypotheses. This can be
        called from the criterion's forward.
        """
        if 'includes_reference' in sample:
            return
        sample['includes_reference'] = True
        target = sample['target']
        pad_idx = self.target_dictionary.pad()
        for i, hypos_i in enumerate(sample['hypos']):
            # insert reference as first hypothesis
            ref = utils.strip_pad(target[i, :], pad_idx)
            hypos_i.insert(0, {
                'tokens': ref,
                'score': None,
            })

    def get_new_sample_for_hypotheses(self, orig_sample):
        """
        Extract hypotheses from *orig_sample* and return a new collated sample.
        """
        ids = orig_sample['id'].tolist()
        pad_idx = self.source_dictionary.pad()
        samples = [
            {
                'id': ids[i],
                'source': utils.strip_pad(orig_sample['net_input']['src_tokens'][i, :], pad_idx),
                'target': hypo['tokens'],
            }
            for i, hypos_i in enumerate(orig_sample['hypos'])
            for hypo in hypos_i
        ]
        return language_pair_dataset.collate(
            samples, pad_idx=pad_idx, eos_idx=self.source_dictionary.eos(),
            left_pad_source=self.args.left_pad_source, left_pad_target=self.args.left_pad_target,
            sort=False,
        )

    def get_sequence_scorer(self, scorer):
        if scorer not in self._scorers:
            tgt_dict = self.target_dictionary
            if scorer == 'bleu':
                self._scorers[scorer] = BleuScorer(
                    tgt_dict, bpe_symbol=self.args.seq_remove_bpe,
                )
            elif scorer == "bert_score":
                self._scorers[scorer] = BertScoreScorer(tgt_dict, bpe_symbol=self.args.seq_remove_bpe,
                                                        model=self.args.bert_score_model,
                                                        model_type=self.args.bert_score_model_type,
                                                        trg_lang=self.args.target_lang,
                                                        model_device=self.args.bert_score_model_device_id)
            else:
                raise ValueError('Unknown sequence scorer {}'.format(scorer))
        return self._scorers[scorer]

    def get_reward(self, sample, scorer=None, discount_factor=0.95, sentence_level=False):
        
        if scorer is None:
            scorer = self.get_sequence_scorer(self.args.seq_scorer)

        bsz = len(sample['hypos'])
        target = sample['target'].int()
        pad_idx = self.target_dictionary.pad()

        max_len = max([len(hypos[0]['tokens']) for hypos in sample['hypos']])
        #print(max_len)

        if sentence_level:
            reward = torch.zeros(bsz, requires_grad=True).to(sample['target'].device)
        else:
            reward = torch.zeros(bsz, max_len, requires_grad=True).to(sample['target'].device)

        for i, hypos_i in enumerate(sample['hypos']):
            ref = utils.strip_pad(target[i, :], pad_idx).cpu()
            ref = scorer.preprocess_ref(ref)

            if sentence_level:
                reward[i] = scorer.get_reward(ref, scorer.preprocess_hypo(hypos_i[0]))
                continue

            # choose best 
            hypos_i = hypos_i[0]

            for j in range(len(hypos_i['tokens'])):
                # scorer.add_ref_hyp(ref, scorer.preprocess_hypo(hypo))
                
                hypos_ij = deepcopy(hypos_i)
                hypos_ij['tokens'] = hypos_ij['tokens'][:j + 1]
                reward[i, j] = scorer.get_reward(ref, scorer.preprocess_hypo(hypos_ij))
                
                if j > 0:
                    reward[i, j] -= reward[i, j - 1]
                    reward[i, j] *= discount_factor ** j
        
        #print(reward)
        #print(reward.size())
        #print('exiting')
        return scorer.postprocess_costs(reward)
                
    def get_costs(self, sample, scorer=None):
        """Get costs for hypotheses using the specified *scorer*."""
        if scorer is None:
            scorer = self.get_sequence_scorer(self.args.seq_scorer)

        bsz = len(sample['hypos'])
        nhypos = len(sample['hypos'][0])
        target = sample['target'].int()
        pad_idx = self.target_dictionary.pad()

        costs = torch.zeros(bsz, nhypos).to(sample['target'].device)
        for i, hypos_i in enumerate(sample['hypos']):
            ref = utils.strip_pad(target[i, :], pad_idx).cpu()
            ref = scorer.preprocess_ref(ref)
            for j, hypo in enumerate(hypos_i):
                # scorer.add_ref_hyp(ref, scorer.preprocess_hypo(hypo))
                costs[i, j] = scorer.get_cost(ref, scorer.preprocess_hypo(hypo))
        # all_costs = scorer.get_cost()

        # ind = 0
        # for i, hypos_i in enumerate(sample['hypos']):
        #     for j, hypo in enumerate(hypos_i):
        #         costs[i, j] = all_costs[ind]
        #         ind += 1

        return scorer.postprocess_costs(costs)
