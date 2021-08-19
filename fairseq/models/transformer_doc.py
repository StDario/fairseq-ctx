# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from _collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    # FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import conv_tbc, sinusoidal_positional_embedding as spe

from fairseq.models.transformer import TransformerModel, base_architecture, TransformerEncoder, TransformerDecoder, \
    Embedding, DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS, TransformerDecoderLayer, PositionalEmbedding

from fairseq.modules import MultiheadAttention, LayerNorm


class ConvolutionalCompressionNetwork(nn.Module):

    def __init__(self, embed_dim, kernel_window=5, stride=5, max_ctx_sentences=10):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_window, stride=stride, padding=0, groups=1)
        self.relu = nn.ReLU()
        # self.pool = nn.MaxPool1d(kernel_size=5)
        self.pool = nn.AvgPool1d(kernel_size=5)
        self.embed_sentence_positions = nn.Embedding(max_ctx_sentences, embed_dim)

    def forward(self, x):
        bs = x.size(1)
        dim = x.size(3)

        positions = x.size(2) - 1 - torch.arange(x.size(2)).cuda()
        positions = self.embed_sentence_positions(positions)
        x = x.detach()
        x += positions
        x = x.transpose(0, 1).reshape(bs, -1, dim).transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2).transpose(0, 1)
        return x

class SimpleConvolutionalCompressionNetwork(nn.Module):

    def __init__(self, embed_dim, kernel_window=10, stride=2, pool_window=10, use_pool_pos_emb=False, use_attn=True, two_conv=False):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_window, stride=stride, padding=0, groups=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=pool_window)
        if use_pool_pos_emb:
            self.pooled_positional_embedding = spe.SinusoidalPositionalEmbedding(embed_dim, 0)
        self.use_pool_pos_emb = use_pool_pos_emb
        self.use_attn = use_attn

        if use_attn:
            self.attn = MultiheadAttention(embed_dim, 8, dropout=0.1, encoder_decoder_attention=False)
            self.layer_norm_attn = LayerNorm(embed_dim, export=False)

        self.two_conv = two_conv
        if two_conv:
            self.conv2 = nn.Conv1d(embed_dim, embed_dim, int(kernel_window / 2), stride=int(stride / 2), padding=0, groups=1)
            self.pool2 = nn.AvgPool1d(kernel_size=int(pool_window / 2))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        if self.two_conv:
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool2(x)

        x = x.transpose(1, 2).transpose(0, 1)

        if self.use_pool_pos_emb:
            pos = torch.arange(x.size(0)).unsqueeze(1).expand(x.size(0), x.size(1)).cuda()
            pos_emb = self.pooled_positional_embedding(pos)
            x += pos_emb

        if self.use_attn:
            out, _ = self.attn(query=x, key=x, value=x, key_padding_mask=None, incremental_state=None, static_kv=True, need_weights=False)
            out = F.dropout(out, p=0.2, training=self.training)
            x = x + out
            x = self.layer_norm_attn(x)

        return x


class AttentionalCompressionNetwork(nn.Module):

    def __init__(self, embed_dim, max_ctx_sentences=10, compression_rate=100, concat_sent_pos_emb=True):
        super().__init__()
        sent_pos_emb_dim = 8 if concat_sent_pos_emb else embed_dim
        self.embed_sentence_positions = nn.Embedding(max_ctx_sentences, sent_pos_emb_dim)
        self.compression_embed = nn.Embedding(compression_rate, embed_dim)
        nn.init.normal_(self.compression_embed.weight, mean=0, std=embed_dim ** -0.5)

        kv_dim = embed_dim + 8 if concat_sent_pos_emb else embed_dim
        self.attn = MultiheadAttention(
            embed_dim, 8, kdim=kv_dim, vdim=kv_dim,
            dropout=0.1, encoder_decoder_attention=True
        )

        self.sent_attn = MultiheadAttention(
            embed_dim, 8,
            dropout=0.1, encoder_decoder_attention=True
        )

        self.dropout = 0.1
        self.concat_sent_pos_emb = concat_sent_pos_emb

        # export = getattr(args, 'char_inputs', False)
        self.layer_norm = LayerNorm(embed_dim, export=False)
        self.layer_norm_sent = LayerNorm(embed_dim, export=False)
        self.compression_rate = compression_rate
        self.max_ctx_sentences = max_ctx_sentences
        self.embed_scale = math.sqrt(embed_dim)

    def forward(self, x, x_mask, x_sent, x_sent_mask):
        bs = x.size(1)
        dim = x.size(3)

        compression_positions = torch.arange(self.compression_rate).cuda()
        compression_positions = self.compression_embed(compression_positions) # * self.embed_scale
        compression_positions = compression_positions.unsqueeze(1).expand(self.compression_rate, bs, dim)

        positions = x.size(2) - 1 - torch.arange(x.size(2)).cuda()
        positions = self.embed_sentence_positions(positions)
        # residual = x
        x = x.detach()

        if self.concat_sent_pos_emb:
            x = torch.cat([x, positions.expand(x.size(0), bs, x.size(2), positions.size(1))], dim=-1)
            dim += 8
        else:
            x += positions
        x = x.transpose(0, 1).reshape(bs, -1, dim).transpose(0, 1)

        x_mask = x_mask.detach()
        x_mask = x_mask.transpose(0, 1).reshape(-1, bs * self.max_ctx_sentences).transpose(0, 1).reshape(bs, -1)

        x_sent = x_sent.detach()
        x_sent_mask = x_sent_mask.detach() if x_sent_mask is not None else None

        out, _ = self.sent_attn(
            query=compression_positions,
            key=x_sent,
            value=x_sent,
            key_padding_mask=x_sent_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=False,
        )

        out = F.dropout(out, p=self.dropout, training=self.training)
        out = compression_positions + out
        out = self.layer_norm_sent(out)

        residual = out

        out, _ = self.attn(
            query=out,
            key=x,
            value=x,
            key_padding_mask=x_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=False,
        )
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = residual + out
        out = self.layer_norm(out)

        return out

class SimpleAttentionalCompressionNetwork(nn.Module):

    def __init__(self, embed_dim, max_ctx_sentences=10, compression_rate=100):
        super().__init__()
        self.compression_embed = nn.Embedding(compression_rate, embed_dim)
        nn.init.normal_(self.compression_embed.weight, mean=0, std=embed_dim ** -0.5)

        self.attn = MultiheadAttention(
            embed_dim, 8, # kdim=kv_dim, vdim=kv_dim,
            dropout=0.2, encoder_decoder_attention=True
        )

        self.sent_attn = MultiheadAttention(
            embed_dim, 8,
            dropout=0.1, encoder_decoder_attention=True
        )

        self.dropout = 0.2

        # export = getattr(args, 'char_inputs', False)
        self.layer_norm = LayerNorm(embed_dim, export=False)
        self.layer_norm_sent = LayerNorm(embed_dim, export=False)
        self.compression_rate = compression_rate
        self.embed_scale = math.sqrt(embed_dim)

    def forward(self, x, x_mask, x_sent, x_sent_mask):

        bs = x.size(0)
        dim = x.size(2)

        compression_positions = torch.arange(self.compression_rate).cuda()
        compression_positions = self.compression_embed(compression_positions) * self.embed_scale
        compression_positions = compression_positions.unsqueeze(1).expand(self.compression_rate, bs, dim)

        x = x.detach()

        # x = x.transpose(0, 1).reshape(bs, -1, dim).transpose(0, 1)

        # x_mask = x_mask.detach()
        # x_mask = x_mask.transpose(0, 1).reshape(-1, bs * self.max_ctx_sentences).transpose(0, 1).reshape(bs, -1)

        out, _ = self.attn(
            query=compression_positions,
            key=x,
            value=x,
            key_padding_mask=x_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=False,
        )
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = compression_positions + out
        out = self.layer_norm(out)

        return out


class ReconstructCompressionNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, y, doc_encoder_out, doc_encoder_out_compressed, attn, doc_enc_out_mask):

        bs = y.size(1)
        dim = doc_encoder_out.size(3)
        max_len = doc_encoder_out.size(0)

        sent_encoder_out = y.detach()
        doc_encoder_out = doc_encoder_out.detach()
        # doc_encoder_out_compressed = doc_encoder_out_compressed.detach()
        attn = copy.deepcopy(attn)

        doc_encoder_out = doc_encoder_out.transpose(0, 1).reshape(bs, -1, dim).transpose(0, 1)
        doc_enc_out_mask = doc_enc_out_mask.detach()
        doc_enc_out_mask = doc_enc_out_mask.reshape(bs, -1, max_len).transpose(1, 2).reshape(bs, -1)

        doc_rep, _ = attn(
            query=sent_encoder_out,
            key=doc_encoder_out,
            value=doc_encoder_out,
            key_padding_mask=doc_enc_out_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=False
        )

        doc_rep_comp, _ = attn(
            query=sent_encoder_out,
            key=doc_encoder_out_compressed,
            value=doc_encoder_out_compressed,
            key_padding_mask=None,
            incremental_state=None,
            static_kv=True,
            need_weights=False
        )

        doc_rep = torch.sigmoid(doc_rep)
        doc_rep_comp = torch.sigmoid(doc_rep_comp)

        loss = self.loss(doc_rep, doc_rep_comp).float()
        return loss


class DocEncoder(nn.Module):

    def __init__(self, padding_idx, max_ctx_sentences=10, compression_type='conv', embed_dim=256, reconstruct=False):
        super().__init__()
        self.max_ctx_sentences = max_ctx_sentences
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        if compression_type == 'conv':
            self.compression_network = ConvolutionalCompressionNetwork(embed_dim)
            self.compressed_or_not_embed = nn.Embedding(2, 8)
        elif compression_type == 'conv_new':
            self.compression_network = SimpleConvolutionalCompressionNetwork(embed_dim)
        elif compression_type == 'attn' or compression_type == 'attn_conc' or compression_type == 'attn_gate':
            self.compression_network = AttentionalCompressionNetwork(embed_dim)
            # self.compressed_or_not_embed = nn.Embedding(2, 8)
        elif compression_type == 'attn_new':
            self.compression_network = SimpleAttentionalCompressionNetwork(embed_dim)
        elif compression_type == 'none':
            self.compression_network = None

        self.reconstruct = reconstruct

        if reconstruct:
            # kv_dim = embed_dim if compression_type == 'attn' else embed_dim + 8
            self.reconstruct_network = ReconstructCompressionNetwork()

        if compression_type == 'attn_conc':
            self.compressed_or_not_embed = nn.Embedding(2, 8)

        self.compression_type = compression_type

    def get_compression_emb(self, bs, is_compressed=True):
        compress_emb = torch.zeros(bs, dtype=torch.int64) if is_compressed is False \
            else torch.ones(bs, dtype=torch.int64)
        compress_emb = self.compressed_or_not_embed(compress_emb.cuda())
        return compress_emb

    def forward(self, encoder, bs, **kwargs):

        # (bs * max_ctx_sentences, dim)
        ctx_src_tokens = kwargs['ctx_src_tokens']
        ctx_src_lengths = kwargs['ctx_src_lengths']

        dim = self.embed_dim
        doc_rep = {}
        conc_enc_out = None

        with torch.no_grad():
            doc_encoder_out = encoder(ctx_src_tokens, src_lengths=ctx_src_lengths, only_emb=True, emb_with_pos=False)
            doc_encoder_out_mask = None
         
        if self.compression_network is not None:

            if self.compression_type == 'conv_new':
                doc_compressed = self.compression_network(doc_encoder_out)
                doc_rep['encoder_out'] = doc_compressed
                doc_rep['encoder_padding_mask'] = None
            elif self.compression_type == 'conv':

                last_sent = doc_encoder_out[:, :, -1, :]

                last_sent_padding_mask = kwargs['ctx_src_tokens'][:, -1, :].eq(self.padding_idx)
                if not last_sent_padding_mask.any():
                    last_sent_padding_mask = None

                doc_compressed = self.compression_network(doc_encoder_out[:, :, :-1, :])
                doc_compressed_mask = doc_compressed.new(bs, doc_compressed.size(0)).bool().fill_(False).cuda()

                not_compressed_emb = self.get_compression_emb(bs, False)
                not_compressed_emb = not_compressed_emb.unsqueeze(0).expand(last_sent.size(0),
                                                                            bs, not_compressed_emb.size(1))
                last_sent = torch.cat([last_sent, not_compressed_emb], dim=2)

                compressed_emb = self.get_compression_emb(bs, True)
                compressed_emb = compressed_emb.unsqueeze(0).expand(doc_compressed.size(0), bs, compressed_emb.size(1))
                doc_compressed = torch.cat([doc_compressed, compressed_emb], dim=2)

                doc_rep['encoder_out'] = torch.cat([doc_compressed, last_sent], dim=0)
                doc_rep['encoder_padding_mask'] = torch.cat([doc_compressed_mask, last_sent_padding_mask], dim=1) \
                    if last_sent_padding_mask is not None else None

            elif self.compression_type == 'attn' or self.compression_type == 'attn_gate' or self.compression_type == 'attn_new':
                doc_rep['encoder_out'] = self.compression_network(doc_encoder_out, doc_encoder_out_mask,
                                                                  kwargs['encoder_out']['encoder_out'],
                                                                  kwargs['encoder_out']['encoder_padding_mask'])
                doc_rep['encoder_padding_mask'] = None
            elif self.compression_type == 'attn_conc':
                doc_comp = self.compression_network(doc_encoder_out, doc_encoder_out_mask)
                encoder_out = kwargs['encoder_out']

                doc_comp_mask = doc_comp.new(bs, doc_comp.size(0)).fill_(False).bool().cuda()
                compressed_emb = self.get_compression_emb(bs, True)
                compressed_emb = compressed_emb.unsqueeze(0).expand(doc_comp.size(0), bs, compressed_emb.size(1))
                doc_comp = torch.cat([doc_comp, compressed_emb], dim=2)

                doc_rep['encoder_out'] = doc_comp
                doc_rep['encoder_padding_mask'] = None

                not_compressed_emb = self.get_compression_emb(bs, False)
                not_compressed_emb = not_compressed_emb.unsqueeze(0).expand(encoder_out['encoder_out'].size(0), bs,
                                                                            not_compressed_emb.size(1))
                encoder_out['encoder_out'] = torch.cat([encoder_out['encoder_out'], not_compressed_emb], dim=2)

                conc_enc_out_tensor = torch.cat([doc_comp, encoder_out['encoder_out']], dim=0)

                if encoder_out['encoder_padding_mask'] is None:
                    conc_enc_out_mask = None
                else:
                    conc_enc_out_mask = torch.cat([doc_comp_mask, encoder_out['encoder_padding_mask']], dim=1)

                conc_enc_out = {}
                conc_enc_out['encoder_out'] = conc_enc_out_tensor
                conc_enc_out['encoder_padding_mask'] = conc_enc_out_mask

                not_compressed_emb = self.get_compression_emb(doc_encoder_out.size(1), False)
                not_compressed_emb = not_compressed_emb.unsqueeze(0).unsqueeze(2).expand(doc_encoder_out.size(0),
                                                                                         bs, self.max_ctx_sentences,
                                                                                         not_compressed_emb.size(1))
                doc_encoder_out = torch.cat([doc_encoder_out, not_compressed_emb], dim=3)

        else:
            doc_mask = kwargs['ctx_src_lengths'].eq(0)
            doc_rep['encoder_out'] = doc_encoder_out.mean(axis=0).view(bs, self.max_ctx_sentences, -1).transpose(0, 1)
            doc_rep['encoder_padding_mask'] = doc_mask

        return doc_rep, doc_encoder_out, doc_encoder_out_mask, conc_enc_out

    def forward_reconstruct(self, y, doc_enc, doc_enc_compressed, attn_net, doc_enc_out_mask):
        reconstruct_loss = self.reconstruct_network(y, doc_enc, doc_enc_compressed, attn_net, doc_enc_out_mask)
        return reconstruct_loss


@register_model('transformer_doc')
class TransformerModelDoc(TransformerModel):
    """

    """

    def __init__(self, encoder, decoder, max_ctx_sentences=10, compress='conv', reconstruct_compressed=False):
        super().__init__(encoder, decoder)

        self.encoder_memory = {}
        self.max_ctx_sentences = max_ctx_sentences
        self.doc_encoder = DocEncoder(encoder.padding_idx, max_ctx_sentences, compress,
                                      encoder.embed_tokens.embedding_dim, reconstruct_compressed)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--doc-size', type=int, help='the size of the document model representation')
        parser.add_argument('--compression-type', type=str, default='conv',
                            help='Compression type')
        parser.add_argument('--reconstruct-compressed', type=bool, default=False,
                            help='Add a reconstruction loss for the compressed document representation')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModelDoc(encoder, decoder, args.max_ctx_sentences, args.compression_type, args.reconstruct_compressed)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        doc_kv_dim = None
        if args.compression_type == 'conv':
            doc_kv_dim = args.encoder_embed_dim + 8
        enc_kv_dim = None
        if args.compression_type == 'attn_conc':
            enc_kv_dim = args.encoder_embed_dim + 8
        return TransformerDecoderDoc(args, tgt_dict, embed_tokens, doc_kv_dim=doc_kv_dim, enc_kv_dim=enc_kv_dim)
        # return TransformerDecoder(args, tgt_dict, embed_tokens)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        reconstruct_loss = None
        bs = src_tokens.size(0)

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        kwargs['encoder_out'] = encoder_out

        doc_enc_out, doc_enc_out_raw, doc_enc_out_mask, conc_encoder_out = self.doc_encoder(self.encoder, bs, **kwargs)

        
        del kwargs['encoder_out']

        if conc_encoder_out is None:
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, doc_representation=doc_enc_out,
                                       keep_first_y=self.doc_encoder.reconstruct, **kwargs)
        else:
            decoder_out = self.decoder(prev_output_tokens, encoder_out=conc_encoder_out, doc_representation=None,
                                       keep_first_y=self.doc_encoder.reconstruct, **kwargs)

        if self.doc_encoder.reconstruct:
            reconstruct_loss = self.doc_encoder.forward_reconstruct(decoder_out[1]['first_y'],
                                                                    doc_enc_out_raw,
                                                                    # doc_compressed,
                                                                    doc_enc_out['encoder_out'],
                                                                    self.decoder.layers[-1].encoder_attn,
                                                                    doc_enc_out_mask)

        return decoder_out, reconstruct_loss


class TransformerDecoderDoc(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, doc_kv_dim=None, enc_kv_dim=None):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderDocLayer(args, no_encoder_attn, doc_kv_dim=doc_kv_dim, enc_kv_dim=enc_kv_dim)
            for _ in range(args.decoder_layers)
        ])

        # self.doc_projection = nn.Linear(embed_tokens.embedding_dim, int(embed_tokens.embedding_dim))

    def forward(self, prev_output_tokens, encoder_out=None, doc_representation=None, incremental_state=None,
                keep_first_y=False, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            doc_representation:
            keep_first_y: keep first self-attention representation over y

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # doc_representation['encoder_out'] = self.doc_projection(doc_representation['encoder_out'])
        x, extra = self.extract_features(prev_output_tokens, encoder_out, doc_representation, incremental_state,
                                         keep_first_y)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, doc_representation=None, incremental_state=None,
                         keep_first_y=False, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]
        first_y = None

        # decoder layers
        for ind, layer in enumerate(self.layers):
            x, attn, self_attn_x = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                doc_representation['encoder_out'] if doc_representation is not None else None,
                doc_representation['encoder_padding_mask'] if doc_representation is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                return_y=True if ind == 0 and keep_first_y else False
            )
            inner_states.append(x)

            if keep_first_y and ind == 0:
                first_y = self_attn_x

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        additional_ret = {'attn': attn, 'inner_states': inner_states}
        if keep_first_y:
            additional_ret['first_y'] = first_y

        return x, additional_ret


# class TransformerDecoderDocLayer(nn.Module):
class TransformerDecoderDocLayer(TransformerDecoderLayer):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, doc_kv_dim=None, enc_kv_dim=None):
        super().__init__(args, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                         enc_kv_dim=enc_kv_dim)

        if args.compression_type != 'attn_conc':
            if doc_kv_dim is None:
                doc_kv_dim = self.embed_dim

            self.doc_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads, kdim=doc_kv_dim, vdim=doc_kv_dim,
                dropout=args.attention_dropout, encoder_decoder_attention=True
            )
            export = getattr(args, 'char_inputs', False)
            self.doc_layer_norm = LayerNorm(self.embed_dim, export=export)

        if args.compression_type == 'attn_gate' or args.compression_type == 'conv_new' or args.compression_type == 'attn_new':
            self.gate_lin1 = nn.Linear(self.embed_dim, self.embed_dim)
            self.gate_lin2 = nn.Linear(self.embed_dim, self.embed_dim)
            self.gate_norm = LayerNorm(self.embed_dim, export=export)
        else:
            self.gate_lin1 = None
            self.gate_lin2 = None

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        doc_representation=None,
        doc_representation_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        return_y=False
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if return_y:
            self_attn_x = x
        else:
            self_attn_x = None

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        if doc_representation is not None:
            residual = x
            x1 = self.maybe_layer_norm(self.doc_layer_norm, x, before=True)

            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.doc_attn._set_input_buffer(incremental_state, saved_state)

            x1, _ = self.doc_attn(
                query=x1,
                key=doc_representation,
                value=doc_representation,
                key_padding_mask=doc_representation_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn)
            )

            x1 = F.dropout(x1, p=self.dropout, training=self.training)

            x1 = residual + x1
            x1 = self.maybe_layer_norm(self.doc_layer_norm, x1, after=True)
        else:
            x1 = x

        if self.gate_lin1 is None:
            x = x1
        else:
            z = self.gate_lin1(x1) + self.gate_lin2(x)
            z = F.relu(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            # x = z * x + (1 - z) * x1
            x_gate = z * x + (1 - z) * x1
            x = x + x_gate
            x = self.gate_norm(x)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state

        return x, attn, self_attn_x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


@register_model_architecture('transformer_doc', 'transformer_doc')
def transformer_doc(args):
    args.doc_size = getattr(args, 'doc_size', args.encoder_embed_dim)
    base_architecture(args)

