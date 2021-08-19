# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset

MAX_CTX_LENGTH = 100

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, sort=True, src_dataset=None, tgt_dataset=None, max_ctx_sentences=10, is_ctx_in_tokens=False, max_ctx_length=2000
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_ctx(key, left_pad, move_eos_to_beginning=False, is_ctx_in_tokens=False):
        # return data_utils.collate_tokens(
        #     [s[key][i] for s in samples for i in range(10)],
        #     pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        # )

        dataset = src_dataset if key == 'ctx_source' else tgt_dataset

        if not is_ctx_in_tokens:
            # -1 means no context sentence
            #collate_input = [dataset[i][0][:MAX_CTX_LENGTH] if i != -1 else torch.LongTensor([5, eos_idx]) for s in samples for i in s[key]]
            collate_input = []
            for s in samples:
                ctx_samples = []
                for i in s[key]:
                    ctx_samples.append(torch.cat([dataset[i][0][:-1], torch.LongTensor([5])]))
                if len(ctx_samples) > 0:
                    ctx_samples = torch.cat(ctx_samples)
                    ctx_samples = torch.cat([ctx_samples[:-1], torch.LongTensor([eos_idx])])[-max_ctx_length:]
                else:
                    ctx_samples = torch.LongTensor([5, eos_idx])
                #print(ctx_samples)
                #print(ctx_samples.size())
                collate_input.append(ctx_samples)
        else:
            collate_input = [s[key] for s in samples]
        return data_utils.collate_tokens(
            collate_input,
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    # print([s['id'] for s in samples])
    # print([s['ctx_source'] for s in samples])
    # exit()
    src_tokens = merge('source', left_pad=left_pad_source)
    ctx_src_tokens = merge_ctx('ctx_source', left_pad=left_pad_source, is_ctx_in_tokens=is_ctx_in_tokens)
    #ctx_src_tokens = ctx_src_tokens.reshape(src_tokens.size(0), max_ctx_sentences, -1)

    ctx_trg_tokens = None
    ctx_trg_lengths = None


    if samples[0].get('ctx_target', None) is not None:
        ctx_trg_tokens = merge_ctx('ctx_target', left_pad=left_pad_source, is_ctx_in_tokens=is_ctx_in_tokens)
        # ctx_trg_tokens = ctx_trg_tokens.reshape(src_tokens.size(0), max_ctx_sentences, -1)

    id = src_tokens.new([s['id'] for s in samples])
    src_lengths = src_tokens.new([s['source'].numel() for s in samples])

    if not is_ctx_in_tokens:
        # ctx_src_lengths = ctx_src_tokens.new([src_dataset[i][0][:MAX_CTX_LENGTH].numel() if i != -1 else 1 for s in samples for i in s['ctx_source']])
        ctx_src_lengths = ctx_src_tokens.new([ctx_s.size(0) for ctx_s in ctx_src_tokens])
    else:
        #ctx_src_lengths = ctx_src_tokens.new([ctx_s.numel() for s in samples for ctx_s in s['ctx_source']])
        ctx_src_lengths = ctx_src_tokens.new([ctx_s.size(0) for ctx_s in ctx_src_tokens])

    #ctx_src_lengths = ctx_src_lengths.reshape(src_tokens.size(0), max_ctx_sentences)

    #print(src_tokens.size())
    #print(ctx_src_tokens.size())
    #print(src_lengths.size())
    #print(ctx_src_lengths.size())
    #exit()

    if samples[0].get('ctx_target', None) is not None:
        if not is_ctx_in_tokens:
            #ctx_trg_lengths = ctx_trg_tokens.new([tgt_dataset[i][0][:MAX_CTX_LENGTH].numel() if i != -1 else 1 for s in samples for i in s['ctx_target']])
            ctx_trg_lengths = ctx_trg_tokens.new([ctx_t.size(0) for ctx_t in ctx_trg_tokens])
        else:
            #ctx_trg_lengths = ctx_trg_tokens.new([ctx_s.numel() for s in samples for ctx_s in s['ctx_target']])
            ctx_trg_lengths = ctx_trg_tokens.new([ctx_t.size(0) for ctx_t in ctx_trg_tokens])
        #ctx_trg_lengths = ctx_trg_lengths.reshape(src_tokens.size(0), max_ctx_sentences)

    if sort:
        # sort by descending source length
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        ctx_src_tokens = ctx_src_tokens.index_select(0, sort_order)

        if ctx_trg_tokens is not None:
            ctx_trg_tokens = ctx_trg_tokens.index_select(0, sort_order)
        ctx_src_lengths = ctx_src_lengths.index_select(0, sort_order)
        if ctx_trg_lengths is not None:
            ctx_trg_lengths = ctx_trg_lengths.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        if sort:
            target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            if sort:
                prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'ctx_src_tokens': ctx_src_tokens,
            'ctx_trg_tokens': ctx_trg_tokens,
            'ctx_src_lengths': ctx_src_lengths,
            'ctx_trg_lengths': ctx_trg_lengths
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        max_ctx_sentences=10, is_ctx_in_tokens=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.max_ctx_sentences = max_ctx_sentences
        self.is_ctx_in_tokens = is_ctx_in_tokens

    def __getitem__(self, index):
        tgt_item = self.tgt[index][0] if self.tgt is not None else None
        src_item = self.src[index][0]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            # if self.tgt and self.tgt[index][-1] != eos:
            #     tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
            if tgt_item and tgt_item[-1] != eos:
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            # if self.src[index][-1] == eos:
            #     src_item = self.src[index][:-1]
            if src_item[-1] == eos:
                src_item = src_item[:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'ctx_source': self.src[index][1],
            'ctx_target': self.tgt[index][1] if self.tgt is not None else None
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, src_dataset=self.src, tgt_dataset=self.tgt,
            max_ctx_sentences=self.max_ctx_sentences, is_ctx_in_tokens=self.is_ctx_in_tokens
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
