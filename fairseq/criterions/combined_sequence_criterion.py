import math
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqSequenceCriterion, register_criterion


@register_criterion('sequence_risk_combined')
class SequenceRiskCriterionCombined(FairseqSequenceCriterion):

    # def __init__(self, args, dst_dict, token_criterion, sequence_criterion, alpha):
    def __init__(self, args, task):
        super().__init__(args, task)

        from fairseq.tasks.translation_struct import TranslationStructuredPredictionTask
        if not isinstance(task, TranslationStructuredPredictionTask):
            raise Exception(
                'sequence_risk criterion requires `--task=translation_struct`'
            )

        self.token_criterion = task.build_token_level_criterion(args)
        self.sequence_criterion = task.build_sequence_level_criterion(args)
        self.alpha = args.seq_combined_loss_alpha

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--token-level-criterion', type=str,
                            help='Token-level criterion')
        parser.add_argument('--sequence-level-criterion', type=str,
                            help='Sequence-level criterion')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--normalize-costs', action='store_true',
                            help='normalize costs within each hypothesis')
        parser.add_argument('--seq-combined-loss-alpha', type=float, default=0,
                            help='Hyper-paramter for controlling token and sequence loss')

    # def prepare_sample_and_hypotheses(self, model, sample, hypos):
    #     """Apply criterion-specific modifications to the given sample/hypotheses."""
    #     # compute token-level loss (unnormalized)
    #     sample['token_criterion_out'] = self.token_criterion(model, sample)

        # then prepare sample for sequence-level criterion
        # return self.sequence_criterion.prepare_sample_and_hypotheses(model, sample, hypos)

    def get_net_output(self, model, sample):
        return self.sequence_criterion.get_net_output(model, sample)

    def forward(self, model, sample, reduce=True):
        """Compute the sequence-level loss for the given hypotheses.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # get token-level loss
        # net_output = model(**sample['net_input'])
        # tok_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # tok_sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        # previous
        # tok_loss, tok_sample_size, _ = sample['token_criterion_out']
        
        # changed
        tok_loss, tok_sample_size, tok_logging_output, net_enc_output = self.token_criterion(model, sample, reduce)
        # print(self.alpha)
        # print(tok_loss)
        # print(tok_sample_size)

        # compute sequence-level loss
        
        # changed
        sample['net_enc_output'] = net_enc_output
        
        seq_loss, seq_sample_size, seq_logging_output = \
                self.sequence_criterion.forward(model, sample, reduce)
        # print(seq_loss)
        # print(seq_sample_size)
        # print(seq_logging_output)

        # First normalize the loss using the current sample's size.
        # Later, normalize again by the number of replicas.

        loss = self.alpha  * tok_loss / tok_sample_size + \
                   (1 - self.alpha) * seq_loss.sum() / seq_sample_size

        loss = loss.sum()
        sample_size = 1  # normalize gradients by the number of replicas

        seq_logging_output.update({
            # 'loss': loss.data[0],
            'loss': utils.item(loss.data),
            'sample_size': sample_size,
            'tok_loss': utils.item(tok_loss.data.sum().data),
            'tok_sample_size': tok_sample_size,
            'seq_loss': utils.item(seq_loss.data.sum().data),
            'seq_sample_size': seq_sample_size,
        })
        return loss, sample_size, seq_logging_output

    # @staticmethod
    # def aggregate_logging_outputs(logging_outputs):
    #     """Aggregate logging outputs from data parallel training."""
    #     ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
    #     nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
    #     sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
    #     return {
    #         'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
    #         'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
    #         'ntokens': ntokens,
    #         'nsentences': nsentences,
    #         'sample_size': sample_size,
    #     }

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        num_costs = sum(log.get('num_cost', 0) for log in logging_outputs)
        tok_sample_size = sum(log.get('tok_sample_size', 0) for log in logging_outputs)
        seq_sample_size = sum(log.get('seq_sample_size', 0) for log in logging_outputs)
        agg_outputs = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'tok_loss': sum(log.get('tok_loss', 0) for log in logging_outputs) / tok_sample_size / math.log(2),
            'seq_loss': sum(log.get('seq_loss', 0) for log in logging_outputs) / seq_sample_size,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
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

    # @staticmethod
    # def aggregate_logging_outputs(logging_outputs):
    #     """Aggregate logging outputs from data parallel training."""
    #     sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
    #     tok_sample_size = sum(log.get('tok_sample_size', 0) for log in logging_outputs)
    #     seq_sample_size = sum(log.get('seq_sample_size', 0) for log in logging_outputs)
    #     return {
    #         'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size ,
    #         'tok_loss': sum(log.get('tok_loss', 0) for log in logging_outputs) / tok_sample_size / math.log(2),
    #         'seq_loss': sum(log.get('seq_loss', 0) for log in logging_outputs) / seq_sample_size,
    #     }