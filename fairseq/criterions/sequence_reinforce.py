# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from fairseq import utils
from fairseq.criterions import FairseqSequenceCriterion, register_criterion

class BaselineEstimator(nn.Module):

    def __init__(self, input_size):
        super(BaselineEstimator, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, input, mean=False):

        input = input.detach()

        if mean:
            input = input.mean(axis=0)
        out = self.linear(input)
        # out = F.relu(out)
        out = torch.sigmoid(out)
        return out

@register_criterion('sequence_reinforce')
class SequenceReinforceCriterion(FairseqSequenceCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        from fairseq.tasks.translation_struct import TranslationStructuredPredictionTask
        if not isinstance(task, TranslationStructuredPredictionTask):
            raise Exception(
                'sequence_risk criterion requires `--task=translation_struct`'
            )

        self.baseline = BaselineEstimator(args.encoder_embed_dim)
        self.baseline.half()
        self.baseline_optimizer = optim.SGD(self.baseline.parameters(), lr=0.1)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--normalize-costs', action='store_true',
                            help='normalize costs within each hypothesis')
        # fmt: on

    def forward(self, model, sample, reduce=True, use_enc_for_baseline=False, sentence_level_reward=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        bsz = len(sample['hypos'])
        nhypos = len(sample['hypos'][0])

        # get costs for hypotheses using --seq-scorer (defaults to 1. - BLEU)
        rewards = self.task.get_reward(sample, sentence_level=sentence_level_reward)

        if self.args.normalize_costs:
            unnormalized_costs = rewards.clone()
            max_costs = rewards.max(dim=1, keepdim=True)[0]
            min_costs = rewards.min(dim=1, keepdim=True)[0]
            rewards = (rewards - min_costs) / (max_costs - min_costs).clamp_(min=1e-6)
        else:
            unnormalized_costs = None

        # generate a new sample from the given hypotheses
        new_sample = self.task.get_new_sample_for_hypotheses(sample)
        hypotheses = new_sample['target'].view(bsz, nhypos, -1, 1)
        hypolen = hypotheses.size(2)
        pad_mask = hypotheses.ne(self.task.target_dictionary.pad())
        lengths = pad_mask.sum(dim=2).float()

        net_output, _ = model(**new_sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(bsz, nhypos, hypolen, -1)

        scores = lprobs.gather(3, hypotheses)
        scores *= pad_mask.float()

        if sentence_level_reward:
            scores = scores.sum(dim=2) / lengths

        # if not use_enc_for_baseline:
        rewards = rewards.unsqueeze(1).unsqueeze(3) if not sentence_level_reward else rewards.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # use_enc_for_baseline=True -> baseline_reward.size() = (bsz, 1)
        # use_enc_for_baseline=False -> baseline_reward.size() = (hypolen, bsz, 1)
        base_input = sample['net_enc_output']['encoder_out'] if use_enc_for_baseline else net_output[1]['inner_states'][-1]
        baseline_reward = self.baseline.forward(base_input, mean=use_enc_for_baseline).float()

        if use_enc_for_baseline:
            baseline_reward = baseline_reward.unsqueeze(2).unsqueeze(3).expand_as(rewards).clone()
        else:
            baseline_reward = baseline_reward.view(bsz, 1, hypolen, -1)

        if not sentence_level_reward:
            baseline_reward *= pad_mask.float()

        # advantage = rewards - baseline_reward

        self.baseline_optimizer.zero_grad()
        bl = torch.nn.MSELoss()
        bl.requires_grad = True
        # print(baseline_reward)
        # print(rewards)
        baseline_loss = bl(baseline_reward, rewards)
        # print(baseline_loss)
        baseline_loss.backward(retain_graph=True)
        self.baseline_optimizer.step()

        # print(a[0])

        # eps = np.finfo(np.float32).eps.item()
        # rewards = (rewards - rewards.mean(axis=2).unsqueeze(3)) / (rewards.std(axis=2).unsqueeze(3) + eps)

        loss = (-(rewards - baseline_reward) * scores).sum()
        # loss = (-rewards * scores).sum()

        # print(loss)

        # avg_scores = scores.sum(dim=2) / lengths
        # probs = F.softmax(avg_scores, dim=1).squeeze(-1)
        # loss = (probs * costs).sum()

        sample_size = bsz
        assert bsz == utils.item(rewards.size(dim=0))
        logging_output = {
            'loss': utils.item(loss.data),
            'num_cost': rewards.numel(),
            'ntokens': sample['ntokens'],
            'nsentences': bsz,
            'sample_size': sample_size,
        }

        def add_cost_stats(costs, prefix=''):
            logging_output.update({
                prefix + 'sum_cost': utils.item(costs.sum()),
                prefix + 'min_cost': utils.item(costs.min(dim=1)[0].sum()),
                prefix + 'cost_at_1': utils.item(costs[:, 0].sum()),
            })

        add_cost_stats(rewards)
        if unnormalized_costs is not None:
            add_cost_stats(unnormalized_costs, 'unnormalized_')

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        num_costs = sum(log.get('num_cost', 0) for log in logging_outputs)
        agg_outputs = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }

        def add_cost_stats(prefix=''):
            agg_outputs.update({
                prefix + 'avg_cost': sum(log.get(prefix + 'sum_cost', 0) for log in logging_outputs) / num_costs,
                prefix + 'min_cost': sum(log.get(prefix + 'min_cost', 0) for log in logging_outputs) / nsentences,
                prefix + 'cost_at_1': sum(log.get(prefix + 'cost_at_1', 0) for log in logging_outputs) / nsentences,
            })

        add_cost_stats()
        if any('unnormalized_sum_cost' in log for log in logging_outputs):
            add_cost_stats('unnormalized_')

        return agg_outputs
