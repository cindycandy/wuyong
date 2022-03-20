# coding=utf-8
from __future__ import print_function

import os
from six.moves import xrange as range
import math
from collections import OrderedDict
import numpy as np
import attr
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from asdl.hypothesis import Hypothesis, GenTokenAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, Action
from common.registerable import Registrable
from components.decode_hypothesis import DecodeHypothesis
from components.action_info import ActionInfo
from components.dataset import Batch
from common.utils import update_args, init_arg_parser
from model import nn_utils
from model.attention_util import AttentionUtil
from model.nn_utils import LabelSmoothing
from model.pointer_net import PointerNet
import model.transformer as transformer

@Registrable.register('default_parser')
class Parser(nn.Module):
    """Implementation of a semantic parser

    The parser translates a natural language utterance into an AST defined under
    the ASDL specification, using the transition system described in https://arxiv.org/abs/1810.02720
    """
    def __init__(self, args, vocab, transition_system):
        super(Parser, self).__init__()

        self.args = args
        self.vocab = vocab

        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar
        self.relation_ids = 13
        # Embedding layers

        # source token embedding
        self.src_embed = nn.Embedding(len(vocab.source), args.embed_size)

        # embedding table of ASDL production rules (constructors), one for each ApplyConstructor action,
        # the last entry is the embedding for Reduce action
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 1, args.action_embed_size)

        # embedding table for target primitive tokens
        self.primitive_embed = nn.Embedding(len(vocab.primitive), args.action_embed_size)

        # embedding table for ASDL fields in constructors
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)

        # embedding table for ASDL types
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        nn.init.xavier_normal_(self.src_embed.weight.data)
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.primitive_embed.weight.data)
        nn.init.xavier_normal_(self.field_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)

        # BiLSTMs,仿照rat，实质和tranx的encoder_lstm一样
        self.lstm = nn.LSTM(
            input_size=self.args.embed_size,
            hidden_size=self.args.hidden_size//2,
            bidirectional=True,
            dropout=self.args.dropout
        )
        #自写，为了将query的h和api的h连接后缩回128
        self.hc_linear = torch.nn.Linear(self.args.hidden_size * 2, self.args.hidden_size)
        if args.lstm == 'lstm':
            self.encoder_lstm = nn.LSTM(args.embed_size, int(args.hidden_size / 2), bidirectional=True)

            input_dim = args.action_embed_size  # previous action
            # frontier info
            input_dim += args.action_embed_size * (not args.no_parent_production_embed)
            input_dim += args.field_embed_size * (not args.no_parent_field_embed)
            input_dim += args.type_embed_size * (not args.no_parent_field_type_embed)
            input_dim += args.hidden_size * (not args.no_parent_state)

            input_dim += args.att_vec_size * (not args.no_input_feed)  # input feeding


            self.decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)
        elif args.lstm == 'parent_feed':
            self.encoder_lstm = nn.LSTM(args.embed_size, int(args.hidden_size / 2), bidirectional=True)
            from .lstm import ParentFeedingLSTMCell

            input_dim = args.action_embed_size  # previous action
            # frontier info
            input_dim += args.action_embed_size * (not args.no_parent_production_embed)
            input_dim += args.field_embed_size * (not args.no_parent_field_embed)
            input_dim += args.type_embed_size * (not args.no_parent_field_type_embed)
            input_dim += args.att_vec_size * (not args.no_input_feed)  # input feeding

            self.decoder_lstm = ParentFeedingLSTMCell(input_dim, args.hidden_size)

        elif args.lstm == "attention":
            self.encoder_lstm = transformer.Encoder(
            lambda: transformer.EncoderLayer(
                self.args.hidden_size,
                transformer.MultiHeadedAttentionWithRelations(
                    self.args.num_heads,
                    self.args.hidden_size,
                    self.args.dropout),
                transformer.PositionwiseFeedForward(
                    self.args.hidden_size,
                    self.args.hidden_size * 4,
                    self.args.dropout),
                self.relation_ids,
                self.args.dropout),
            self.args.hidden_size,
            self.args.num_layers)

            input_dim = args.action_embed_size  # previous action
            # frontier info
            input_dim += args.action_embed_size * (not args.no_parent_production_embed)
            input_dim += args.field_embed_size * (not args.no_parent_field_embed)
            input_dim += args.type_embed_size * (not args.no_parent_field_type_embed)
            input_dim += args.hidden_size * (not args.no_parent_state)

            input_dim += args.att_vec_size * (not args.no_input_feed)  # input feeding

            self.decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)
        else:
            raise ValueError('Unknown LSTM type %s' % args.lstm)

        if args.no_copy is False:
            # pointer net for copying tokens from source side
            self.src_pointer_net = PointerNet(query_vec_size=args.att_vec_size, src_encoding_size=args.hidden_size)

            # given the decoder's hidden state, predict whether to copy or generate a target primitive token
            # output: [p(gen(token)) | s_t, p(copy(token)) | s_t]

            self.primitive_predictor = nn.Linear(args.att_vec_size, 2)

        if args.primitive_token_label_smoothing:
            self.label_smoothing = LabelSmoothing(args.primitive_token_label_smoothing, len(self.vocab.primitive), ignore_indices=[0, 1, 2])

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's hidden space

        self.att_src_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)

        self.att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        # bias for predicting ApplyConstructor and GenToken actions
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(transition_system.grammar) + 1).zero_())
        self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab.primitive)).zero_())

        #为新的attention服务
        self.dk = 128
        self.d_ff = 256
        self.Norm = nn.LayerNorm(args.action_embed_size)
        self.src_att_linear = nn.Linear(args.action_embed_size, args.num_heads*self.dk)
        import model.new_transformer as t
        #这里只设置两层layer，之后的第二层layer添加新的注意
        self.new_att_trans = t.Transformer(args.action_embed_size,self.dk,args.num_heads,self.d_ff,2,0.5)

        if args.no_query_vec_to_action_map:
            # if there is no additional linear layer between the attentional vector (i.e., the query vector)
            # and the final softmax layer over target actions, we use the attentional vector to compute action
            # probabilities

            assert args.att_vec_size == args.action_embed_size
            self.production_readout = lambda q: F.linear(q, self.production_embed.weight, self.production_readout_b)
            self.tgt_token_readout = lambda q: F.linear(q, self.primitive_embed.weight, self.tgt_token_readout_b)
        else:
            # by default, we feed the attentional vector (i.e., the query vector) into a linear layer without bias, and
            # compute action probabilities by dot-producting the resulting vector and (GenToken, ApplyConstructor) action embeddings
            # i.e., p(action) = query_vec^T \cdot W \cdot embedding

            #ett_vec_size为256
            self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.embed_size, bias=args.readout == 'non_linear')
            if args.query_vec_to_action_diff_map:
                # use different linear transformations for GenToken and ApplyConstructor actions
                self.query_vec_to_primitive_embed = nn.Linear(args.att_vec_size, args.embed_size, bias=args.readout == 'non_linear')
            else:
                self.query_vec_to_primitive_embed = self.query_vec_to_action_embed
            #这里readout是identity，所以readoutact是返回原值，什么也不做
            self.read_out_act = torch.tanh if args.readout == 'non_linear' else nn_utils.identity
            #乘以rule embedding 相关的emb矩阵，同时b是随机初始化的偏置，下面readout同理，唯一区别在于emb的权重采用不同
            # print("att_vec_size",self.att_vec_size.shape)

            # print("self.production_embed.weight",self.production_embed.weight.shape)
            self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                         self.production_embed.weight, self.production_readout_b)
            self.tgt_token_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_primitive_embed(q)),
                                                        self.primitive_embed.weight, self.tgt_token_readout_b)


        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

    def update_with_relation(self, src_enc, relation=None,lens=None):
        src_enc = src_enc.transpose(0,1)
        # print("\nsrc enc shape",src_enc.shape)
        if relation[0] is None or relation is None:
            relation_t = None
            enc_new = self.encoder_lstm(src_enc, relation_t, mask=None)
        else:
            relation_t = self.new_long_tensor(relation)
            # print("lens",lens,relation_t.shape)
            atten_mask = self.get_attn_mask(lens)
            #这里基本确定要用mask，因此不再作为对比点
            enc_new = self.encoder_lstm(src_enc, relation_t, mask=atten_mask)
        # print("\nafter update enc shape",enc_new.shape)
        # src_enc_new = enc_new[:, :]
        return enc_new

    def get_attn_mask(self,seq_lengths):
        # Given seq_lengths like [3, 1, 2], this will produce
        # [[[1, 1, 1],
        #   [1, 1, 1],
        #   [1, 1, 1]],
        #  [[1, 0, 0],
        #   [0, 0, 0],
        #   [0, 0, 0]],
        #  [[1, 1, 0],
        #   [1, 1, 0],
        #   [0, 0, 0]]]
        # int(max(...)) so that it has type 'int instead of numpy.int64
        max_length, batch_size = int(max(seq_lengths)), len(seq_lengths)
        attn_mask = torch.LongTensor(batch_size, max_length, max_length).fill_(0)
        for batch_idx, seq_length in enumerate(seq_lengths):
            attn_mask[batch_idx, :seq_length, :seq_length] = 1
        return attn_mask.cuda()

    def display_attention(self,candidate, translation, attention, path):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        attention = attention.squeeze(1).cpu().detach().numpy()
        cax = ax.matshow(attention, cmap='bone')
        ax.tick_params(labelsize=15)
        ax.set_xticklabels(candidate)
        ax.set_yticklabels(translation)
        plt.xticks(rotation=90)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()
        plt.savefig(path)
        plt.close()

    def new_relation_update(self,src,relation,alpha,lens):
        src = src.transpose(0,1)
        batch_size,emb_size = src.shape[0],src.shape[1]
        # size = src.shape[-1]
        # print(src.shape,self.args.action_embed_size)
        #这里采用直接指定的分布，后续可以思考一个公式
        # standard_heads = torch.tensor([4,4,3,3,2,2,1,0])
        # data_flow_heads = self.args.num_heads - standard_heads
        relation = self.new_tensor(relation)
        mask = [[0 if t<i else 1 for t in range(max(lens))]for i in lens]
        mask = self.new_tensor(mask)
        result,atts = self.new_att_trans(src,mask,relation)
        return result,atts

    def encode(self, src_tokens, src_sents_len=None, relation=None,related_code=None,unchanged = False):
        """Encode the input natural language utterance
        传

        Args:
            src_tokens: a variable of shape (src_sent_len, batch_size), representing word ids of the input
            src_sents_len: a list of lengths of input source sentences, sorted by descending order

        Returns:
            src_encodings: source encodings of shape (batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: the last hidden state and cell state of the encoder,
                                   of shape (batch_size, hidden_size)
        """
        # print(src_sents_len)
        # unchanged = True
        if self.args.lstm == "lstm":
            # (tgt_query_len, batch_size, embed_size)
            # apply word dropout
            if self.training and self.args.word_dropout:
                mask = Variable(self.new_tensor(src_tokens.size()).fill_(1. - self.args.word_dropout).bernoulli().long())
                src_tokens = src_tokens * mask + (1 - mask) * self.vocab.source.unk_id
            # print("the var shape:",src_tokens.shape)
            src_token_embed = self.src_embed(src_tokens)

            # print("the shape after the emb:", src_token_embed.shape)
            src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len)
            src_encodings, (last_state, last_cell) = self.encoder_lstm(src_token_embed)
            src_encodings, _ = pad_packed_sequence(src_encodings)
            #利用rat的attention更新
            # src_encodings = self.update_with_relation(src_encodings.data, relation,src_sents_len)
            #利用dataflow的方式更新
            # src_encodings,atts = self.new_relation_update(src_encodings.data, relation,5,src_sents_len)
            # print("last state",src_tokens.transpose(0,1)[0],atts[0][0][0].shape,src_sents_len)
            # self.display_attention(src_tokens.transpose(0,1)[0].tolist(),src_tokens.transpose(0,1)[0].tolist(),atts[0][0][0],"m.jpg")
            # src_encodings: (batch_size, tgt_query_len, hidden_size)
            src_encodings = src_encodings.permute(1, 0, 2)
            last_state = torch.cat([last_state[0], last_state[1]], 1)
            last_cell = torch.cat([last_cell[0], last_cell[1]], 1)
            # print(src_encodings.shape, "return one")
        # src_token_embed = packed_src_token_embed.permute(1,0,2)

        # elif unchanged == True:
        #     print("To use new attention")
        #     if self.training and self.args.word_dropout:
        #         mask = Variable(self.new_tensor(src_tokens.size()).fill_(1. - self.args.word_dropout).bernoulli().long())
        #         src_tokens = src_tokens * mask + (1 - mask) * self.vocab.source.unk_id
        #     # print(src_tokens)
        #     src_token_embed = self.src_embed(src_tokens)
        #     packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len)
        #     src_encodings, (last_state, last_cell) = self.lstm(packed_src_token_embed)
        #     src_encodings, _ = pad_packed_sequence(src_encodings)
        #     #在这里改
        #     # src_encodings = self.new_relation_update(src_encodings.data, relation,0.2,self.args.hidden_size)
        #     #记得改回来
        #     src_encodings = self.update_with_relation(src_encodings.data, relation,src_sents_len)
        #     last_state = torch.cat([last_state[0], last_state[1]], 1)
        #     last_cell = torch.cat([last_cell[0], last_cell[1]], 1)
        #     # print("former way finished",src_encodings.shape,last_cell.shape)

        elif self.args.lstm == "attention":
            src_embeds = []
            last_state_batch_a,last_state_batch_b,last_cell_batch_a,last_cell_batch_b = [],[],[],[]
            lens = []
            # print(src_tokens)
            for query,api in zip(src_tokens,related_code):
                lens.append(len(query) + len(api))
                query = Variable(torch.LongTensor(nn_utils.word2id(query,self.vocab.source))).cuda()
                api = Variable(torch.LongTensor(nn_utils.word2id(api,self.vocab.source)).cuda())
                query_emb = self.src_embed(query)
                api_emb = self.src_embed(api)
                #bilstm
                query_enc, (last_state_a, last_cell_a) = self.lstm(torch.unsqueeze(query_emb,1))
                api_enc, (last_state_b, last_cell_b) = self.lstm(torch.unsqueeze(api_emb,1))
                last_state_batch_a.append(last_state_a)
                last_state_batch_b.append(last_state_b)
                last_cell_batch_a.append(last_cell_a)
                last_cell_batch_b.append(last_cell_b)
                # print("h",last_state_a.shape,"c",last_cell_a.shape)
                # h = torch.cat(last_state_a.squeeze().resize(1,128))
                src_embed = torch.cat((query_enc,api_enc),0)
                # print(src_embed.shape)
                src_embeds.append(src_embed)
            #处理h和c，完全实现batch化
            last_state_batch_a = torch.cat(last_state_batch_a,-2)
            last_state_batch_b = torch.cat(last_state_batch_b, -2)
            last_cell_batch_a = torch.cat(last_cell_batch_a, -2)
            last_cell_batch_b = torch.cat(last_cell_batch_b, -2)
            #连接h，c。 [2,4,64] -> [1,4,128] -> [1,4,256]
            last_state = torch.cat((last_state_batch_a.view(1,-1,self.args.hidden_size),last_state_batch_b.view(1,-1,self.args.hidden_size)),-1)
            last_cell = torch.cat((last_cell_batch_a.view(1, -1, self.args.hidden_size),
                                    last_cell_batch_b.view(1, -1, self.args.hidden_size)), -1)

            last_state = self.hc_linear(last_state)
            last_cell = self.hc_linear(last_cell)
            src_embs_pad = nn.utils.rnn.pad_sequence(src_embeds,batch_first=True).squeeze(-2)
            # print(src_embs_pad.shape,"pad secceed")
            last_state = last_state.squeeze(0)
            last_cell = last_cell.squeeze(0)
            # print(last_cell.shape, "squeeze too much")
            # src_encodings: (tgt_query_len, batch_size, hidden_size)
            # src_enc_new: (batch_size,query_len,embeding_size)
            # print(lens,"lens")
            src_encodings = self.update_with_relation(src_embs_pad.data.transpose(0,1), relation,lens)
            # print(src_encodings.shape,"return one")


        #before
        # # (batch_size, hidden_size * 2)
        # last_state = torch.cat([last_state[0], last_state[1]], 1)
        # last_cell = torch.cat([last_cell[0], last_cell[1]], 1)
        atts = []

        return src_encodings, (last_state, last_cell),atts
    # def apply(self,fn):
    #     return attr.evolve(self, ps=torch.nn.utils.rnn.PackedSequence(
    #             fn(self.ps.data), self.ps.batch_sizes))
    def init_decoder_state(self, enc_last_state, enc_last_cell):
        """Compute the initial decoder hidden state and cell state"""

        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

    def encode_(self,batch=None,src_sent=None,lens=None,relation=None,related_code=None,unchanged=False,parse=False):
        if self.args.lstm == "attention":
            # print("*****************", batch.relation)
            # if batch.relation[0] is None:
            #     pass
            #     # print("the relation is none")
            if self.args.mod == "origin":
                    src_encodings, (last_state, last_cell) = self.encode(batch.src_sents_var, batch.src_sents_len, batch.relation)
            elif self.args.mod == "hard":
                #用加了bi的模型
                if unchanged == False:
                    if parse == False:
                        src_encodings, (last_state, last_cell) = self.encode(batch.src_query, batch.src_sents_len,
                                                                         batch.relation_var, batch.related_code)
                    else:
                        src_encodings, (last_state, last_cell) = self.encode(src_sent, lens, relation,related_code)
                #如果要测试之前的方法，需要改两个参数，一个是下面的encode第一个参数改为batch.src_sents_var，第二个是encode内的unchanged改为True
                #如果要测试新的方法，第一个参数是batch.src_query,False
                elif unchanged == True:

                    if parse == False:
                        src_encodings, (last_state, last_cell) = self.encode(batch.src_sents_var, batch.src_sents_len, batch.relation_var,unchanged=True)
                    else:
                        src_encodings, (last_state, last_cell) = self.encode(src_sent, lens, relation,unchanged=True)

        else:
            # print("before encode",batch.src_sents_var.shape)
            src_encodings, (last_state, last_cell),_ = self.encode(batch.src_sents_var, batch.src_sents_len,relation=batch.relation_var)
            # print("after encode",src_encodings.shape,last_state.shape,last_cell.shape)
        return src_encodings, (last_state, last_cell)

    def score(self, examples, return_encode_state=False):
        """Given a list of examples, compute the log-likelihood of generating the target AST

        Args:
            examples: a batch of examples
            return_encode_state: return encoding states of input utterances
        output: score for each training example: Variable(batch_size)
        """
        batch = Batch(examples, self.grammar, self.vocab, copy=self.args.no_copy is False, cuda=self.args.cuda)
        # print("the start shape", batch.src_sents_var)
        # src_encodings: (batch_size, src_sent_len, hidden_size * 2)
        # (last_state, last_cell, dec_init_vec): (batch_size, hidden_size)
        # print("it is the score method, is used before the encode method")
        #2222
        src_encodings, (last_state, last_cell) = self.encode_(batch,unchanged=True,parse=False)
        # if self.args.lstm == "attention":
        #     # print("*****************", batch.relation)
        #     if batch.relation[0] is None:
        #         pass
        #         # print("the relation is none")
        #     if self.args.mod == "origin":
        #         src_encodings, (last_state, last_cell) = self.encode(batch.src_sents_var, batch.src_sents_len, batch.relation)
        #     elif self.args.mod == "hard":
        #         #如果要测试之前的方法，需要改两个参数，一个是下面的encode第一个参数改为batch.src_sents_var，第二个是encode内的unchanged改为True
        #         #如果要测试新的方法，第一个参数是batch.src_query,False
        #         src_encodings, (last_state, last_cell) = self.encode(batch.src_query, batch.src_sents_len, batch.relation_var,batch.related_code,batch.vocab)
        # else:
        #     # print("before encode",batch.src_sents_var.shape)
        #     src_encodings, (last_state, last_cell) = self.encode(batch.src_sents_var, batch.src_sents_len)
        #     # print("after encode",src_encodings.shape,last_state.shape,last_cell.shape)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        # query vectors are sufficient statistics used to compute action probabilities
        # query_vectors: (tgt_action_len, batch_size, hidden_size)

        # if use supervised attention
        if self.args.sup_attention:
            query_vectors, att_prob = self.decode(batch, src_encodings, dec_init_vec)
        else:
            query_vectors = self.decode(batch, src_encodings, dec_init_vec)

        # ApplyRule (i.e., ApplyConstructor) action probabilities
        # (tgt_action_len, batch_size, grammar_size)
        apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)

        # probabilities of target (gold-standard) ApplyRule actions
        # (tgt_action_len, batch_size)

        #gather根据index来进行取值，index是一个同一致的矩阵
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2,
                                           index=batch.apply_rule_idx_matrix.unsqueeze(2)).squeeze(2)

        #### compute generation and copying probabilities

        # (tgt_action_len, batch_size, primitive_vocab_size)
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)

        # print("gen_from_vocab_prob",gen_from_vocab_prob.shape)
        # (tgt_action_len, batch_size)
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=batch.primitive_idx_matrix.unsqueeze(2)).squeeze(2)

        # print("the tgt_primitive_gen_from_vocab_prob",tgt_primitive_gen_from_vocab_prob)
        #no_copy为False
        if self.args.no_copy:
            # mask positions in action_prob that are not used

            if self.training and self.args.primitive_token_label_smoothing:
                # (tgt_action_len, batch_size)
                # this is actually the negative KL divergence size we will flip the sign later
                # tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                #     gen_from_vocab_prob.view(-1, gen_from_vocab_prob.size(-1)).log(),
                #     batch.primitive_idx_matrix.view(-1)).view(-1, len(batch))

                tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                    gen_from_vocab_prob.log(),
                    batch.primitive_idx_matrix)
            else:
                tgt_primitive_gen_from_vocab_log_prob = tgt_primitive_gen_from_vocab_prob.log()

            # (tgt_action_len, batch_size)
            action_prob = tgt_apply_rule_prob.log() * batch.apply_rule_mask + \
                          tgt_primitive_gen_from_vocab_log_prob * batch.gen_token_mask
        else:
            # binary gating probabilities between generating or copying a primitive token
            # (tgt_action_len, batch_size, 2)
            primitive_predictor = F.softmax(self.primitive_predictor(query_vectors), dim=-1)

            # pointer network copying scores over source tokens
            # (tgt_action_len, batch_size, src_sent_len)
            primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask, query_vectors)

            # marginalize over the copy probabilities of tokens that are same
            # (tgt_action_len, batch_size)
            tgt_primitive_copy_prob = torch.sum(primitive_copy_prob * batch.primitive_copy_token_idx_mask, dim=-1)

            # mask positions in action_prob that are not used
            # (tgt_action_len, batch_size)
            action_mask_pad = torch.eq(batch.apply_rule_mask + batch.gen_token_mask + batch.primitive_copy_mask, 0.)

            action_mask = 1. - action_mask_pad.float()

            # (tgt_action_len, batch_size)
            action_prob = tgt_apply_rule_prob * batch.apply_rule_mask + \
                          primitive_predictor[:, :, 0] * tgt_primitive_gen_from_vocab_prob * batch.gen_token_mask + \
                          primitive_predictor[:, :, 1] * tgt_primitive_copy_prob * batch.primitive_copy_mask
            # print(action_prob)
            # print("action_prob batch size", action_prob.shape)
            # avoid nan in log
            action_prob.data.masked_fill_(action_mask_pad.data, 1.e-7)

            action_prob = action_prob.log() * action_mask

        # print("action_prob",action_prob.shape)

        scores = torch.sum(action_prob, dim=0)
        # print("scores",scores.shape)
        # print("the scores and the action_prob",scores)

        returns = [scores]
        if self.args.sup_attention:
            returns.append(att_prob)
        if return_encode_state: returns.append(last_state)
        # print("returns",returns)
        return returns

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_token_mask=None, return_att_weight=False):
        """Perform a single time-step of computation in decoder LSTM

        Args:
            x: variable of shape (batch_size, hidden_size), input
            h_tm1: Tuple[Variable(batch_size, hidden_size), Variable(batch_size, hidden_size)], previous
                   hidden and cell states
            src_encodings: variable of shape (batch_size, src_sent_len, hidden_size * 2), encodings of source utterances
            src_encodings_att_linear: linearly transformed source encodings
            src_token_mask: mask over source tokens (Note: unused entries are masked to **one**)
            return_att_weight: return attention weights

        Returns:
            The new LSTM hidden state and cell state
        """

        # h_t: (batch_size, hidden_size)
        #decoder lstm是一个lstmCell，input每次要求是[batch，dima]，hx和cx要求是[batch，dimb]，初始化是[dima,dimb],输出是hx,cx的更新
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else: return (h_t, cell_t), att_t

    def decode(self, batch, src_encodings, dec_init_vec):
        """Given a batch of examples and their encodings of input utterances,
        compute query vectors at each decoding time step, which are used to compute
        action probabilities

        Args:
            batch: a `Batch` object storing input examples
            src_encodings: variable of shape (batch_size, src_sent_len, hidden_size * 2), encodings of source utterances
            dec_init_vec: a tuple of variables representing initial decoder states

        Returns:
            Query vectors, a variable of shape (tgt_action_len, batch_size, hidden_size)
            Also return the attention weights over candidate tokens if using supervised attention
        """

        batch_size = len(batch)
        args = self.args

        if args.lstm == 'parent_feed':
            h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                    Variable(self.new_tensor(batch_size, args.hidden_size).zero_()), \
                    Variable(self.new_tensor(batch_size, args.hidden_size).zero_())
        else:
            h_tm1 = dec_init_vec

        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        att_vecs = []
        history_states = []
        att_probs = []
        att_weights = []

        for t in range(batch.max_action_num):
            # the input to the decoder LSTM is a concatenation of multiple signals
            # [
            #   embedding of previous action -> `a_tm1_embed`,
            #   previous attentional vector -> `att_tm1`,
            #   embedding of the current frontier (parent) constructor (rule) -> `parent_production_embed`,
            #   embedding of the frontier (parent) field -> `parent_field_embed`,
            #   embedding of the ASDL type of the frontier field -> `parent_field_type_embed`,
            #   LSTM state of the parent action -> `parent_states`
            # ]

            if t == 0:
                x = Variable(self.new_tensor(batch_size, self.decoder_lstm.input_size).zero_(), requires_grad=False)

                # initialize using the root type embedding
                if args.no_parent_field_type_embed is False:
                    offset = args.action_embed_size  # prev_action
                    offset += args.att_vec_size * (not args.no_input_feed)
                    offset += args.action_embed_size * (not args.no_parent_production_embed)
                    offset += args.field_embed_size * (not args.no_parent_field_embed)

                    x[:, offset: offset + args.type_embed_size] = self.type_embed(Variable(self.new_long_tensor(
                        [self.grammar.type2id[self.grammar.root_type] for e in batch.examples])))
            else:
                a_tm1_embeds = []
                for example in batch.examples:
                    # action t - 1
                    if t < len(example.tgt_actions):
                        a_tm1 = example.tgt_actions[t - 1]
                        if isinstance(a_tm1.action, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.action.production]]
                        elif isinstance(a_tm1.action, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.action.token]]
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if args.no_input_feed is False:
                    inputs.append(att_tm1)
                if args.no_parent_production_embed is False:
                    parent_production_embed = self.production_embed(batch.get_frontier_prod_idx(t))
                    inputs.append(parent_production_embed)
                if args.no_parent_field_embed is False:
                    parent_field_embed = self.field_embed(batch.get_frontier_field_idx(t))
                    inputs.append(parent_field_embed)
                if args.no_parent_field_type_embed is False:
                    parent_field_type_embed = self.type_embed(batch.get_frontier_field_type_idx(t))
                    inputs.append(parent_field_type_embed)

                # append history states
                actions_t = [e.tgt_actions[t] if t < len(e.tgt_actions) else None for e in batch.examples]
                if args.no_parent_state is False:
                    parent_states = torch.stack([history_states[p_t][0][batch_id]
                                                 for batch_id, p_t in
                                                 enumerate(a_t.parent_t if a_t else 0 for a_t in actions_t)])

                    parent_cells = torch.stack([history_states[p_t][1][batch_id]
                                                for batch_id, p_t in
                                                enumerate(a_t.parent_t if a_t else 0 for a_t in actions_t)])

                    if args.lstm == 'parent_feed':
                        h_tm1 = (h_tm1[0], h_tm1[1], parent_states, parent_cells)
                    else:
                        inputs.append(parent_states)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, src_encodings,
                                                         src_encodings_att_linear,
                                                         src_token_mask=batch.src_token_mask,
                                                         return_att_weight=True)

            # if use supervised attention
            if args.sup_attention:
                for e_id, example in enumerate(batch.examples):
                    if t < len(example.tgt_actions):
                        action_t = example.tgt_actions[t].action
                        cand_src_tokens = AttentionUtil.get_candidate_tokens_to_attend(example.src_sent, action_t)
                        if cand_src_tokens:
                            att_prob = [att_weight[e_id, token_id] for token_id in cand_src_tokens]
                            if len(att_prob) > 1: att_prob = torch.cat(att_prob).sum()
                            else: att_prob = att_prob[0]
                            att_probs.append(att_prob)

            history_states.append((h_t, cell_t))
            att_vecs.append(att_t)
            att_weights.append(att_weight)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        att_vecs = torch.stack(att_vecs, dim=0)
        if args.sup_attention:
            return att_vecs, att_probs
        else: return att_vecs

    def parse(self, src_sent, relation, related_code=None, context=None, beam_size=5, debug=False,unchanged=True):
        """Perform beam search to infer the target AST given a source utterance

        Args:
            src_sent: list of source utterance tokens
            context: other context used for prediction
            beam_size: beam size

        Returns:
            A list of `DecodeHypothesis`, each representing an AST
        """

        args = self.args
        primitive_vocab = self.vocab.primitive
        T = torch.cuda if args.cuda else torch

        # print("ppppppppppppppppppppppp",src_sent)
        src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, cuda=args.cuda, training=False)

        # relation = [[]]
        # Variable(1, src_sent_len, hidden_size * 2)
        #原先的方法，由于传入encode的东西有区别，这里要想办法加上限制条件Misdirection Arcane
        import time
        if self.args.lstm=="lstm":
            src_encodings, (last_state, last_cell),atts = self.encode(src_sent_var, [len(src_sent)], relation=relation)
            # print("here it is used"
            # if src_sent[0]=="Misdirection":
            #     t = time.time()
            #     print("the attention source",src_sent,atts[0].shape)
            #     for i in range(4):
            #         imgpath = "attention_figures/img"+t.__str__()+i.__str__()+".jpg"
            #         self.display_attention(src_sent,src_sent,atts[0][0][i],imgpath)
            # print("succesfully save the picture")
        else:
            # print("____________________starting parse___________________________")
            #此处是使用旧方法
            if unchanged==True:
                src_encodings, (last_state, last_cell) = self.encode_(src_sent=src_sent_var,lens=[len(src_sent)] , relation=relation,unchanged=unchanged,parse=True)
            else:
                # attention当前有两个模式，一个是旧的，一个是新的。改成旧的要注意三个点，此处为src_sent_var，改成新的要换成[src_sent]
                src_encodings, (last_state, last_cell) = self.encode_(src_sent=[src_sent],lens=None , relation=relation, related_code=[related_code],unchanged=unchanged,parse=True)
        # (1, src_sent_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        if args.lstm == 'parent_feed':
            h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                    Variable(self.new_tensor(args.hidden_size).zero_()), \
                    Variable(self.new_tensor(args.hidden_size).zero_())
        else:
            h_tm1 = dec_init_vec

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        with torch.no_grad():
            hyp_scores = Variable(self.new_tensor([0.]))

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

        t = 0
        hypotheses = [DecodeHypothesis()]
        hyp_states = [[]]
        completed_hypotheses = []

        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, src_encodings_att_linear.size(1), src_encodings_att_linear.size(2))

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).zero_())
                if args.no_parent_field_type_embed is False:
                    offset = args.action_embed_size  # prev_action
                    offset += args.att_vec_size * (not args.no_input_feed)
                    offset += args.action_embed_size * (not args.no_parent_production_embed)
                    offset += args.field_embed_size * (not args.no_parent_field_embed)

                    x[0, offset: offset + args.type_embed_size] = \
                        self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]

                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        if isinstance(a_tm1, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                        elif isinstance(a_tm1, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]

                        a_tm1_embeds.append(a_tm1_embed)
                    else:
                        a_tm1_embeds.append(zero_action_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if args.no_input_feed is False:
                    inputs.append(att_tm1)
                if args.no_parent_production_embed is False:
                    # frontier production
                    frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                    frontier_prod_embeds = self.production_embed(Variable(self.new_long_tensor(
                        [self.grammar.prod2id[prod] for prod in frontier_prods])))
                    inputs.append(frontier_prod_embeds)
                #默认为false，需要加入父类值的emb
                if args.no_parent_field_embed is False:
                    # frontier field
                    frontier_fields = [hyp.frontier_field.field for hyp in hypotheses]
                    frontier_field_embeds = self.field_embed(Variable(self.new_long_tensor([
                        self.grammar.field2id[field] for field in frontier_fields])))

                    inputs.append(frontier_field_embeds)
                #true,不加
                if args.no_parent_field_type_embed is False:
                    # frontier field type
                    frontier_field_types = [hyp.frontier_field.type for hyp in hypotheses]
                    frontier_field_type_embeds = self.type_embed(Variable(self.new_long_tensor([
                        self.grammar.type2id[type] for type in frontier_field_types])))
                    inputs.append(frontier_field_type_embeds)

                # parent states，false，加
                if args.no_parent_state is False:
                    p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                    parent_states = torch.stack([hyp_states[hyp_id][p_t][0] for hyp_id, p_t in enumerate(p_ts)])
                    parent_cells = torch.stack([hyp_states[hyp_id][p_t][1] for hyp_id, p_t in enumerate(p_ts)])

                    if args.lstm == 'parent_feed':
                        h_tm1 = (h_tm1[0], h_tm1[1], parent_states, parent_cells)
                    else:
                        inputs.append(parent_states)

                # x:(5,576)
                x = torch.cat(inputs, dim=-1)


            # #att_t (5,256)
            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             src_token_mask=None)

            # Variable(batch_size, grammar_size)
            # apply_rule_log_prob = torch.log(F.softmax(self.production_readout(att_t), dim=-1))

            #apply_rule_log_prob (5,97) 97是action类别数量
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)
            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            # print("gen_from_vocab_prob",gen_from_vocab_prob)

            if args.no_copy:
                primitive_prob = gen_from_vocab_prob
            else:
                # Variable(batch_size, src_sent_len)
                primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

                # Variable(batch_size, 2)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                # Variable(batch_size, primitive_vocab_size)
                primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob

                # if src_unk_pos_list:
                #     primitive_prob[:, primitive_vocab.unk_id] = 1.e-10

            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = []
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            applyrule_prev_hyp_ids = []

            for hyp_id, hyp in enumerate(hypotheses):
                # generate new continuations
                action_types = self.transition_system.get_valid_continuation_types(hyp)
                # print("action_types", hyp_id, action_types)
                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = self.transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            #这样能取到正常的int值，而不是一个tensor
                            prod_score = apply_rule_log_prob[hyp_id, prod_id].data.item()
                            new_hyp_score = hyp.score + prod_score

                            applyrule_new_hyp_scores.append(new_hyp_score)
                            applyrule_new_hyp_prod_ids.append(prod_id)
                            applyrule_prev_hyp_ids.append(hyp_id)
                    elif action_type == ReduceAction:
                        action_score = apply_rule_log_prob[hyp_id, len(self.grammar)].data.item()
                        new_hyp_score = hyp.score + action_score

                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_new_hyp_prod_ids.append(len(self.grammar))
                        applyrule_prev_hyp_ids.append(hyp_id)
                    else:
                        # GenToken action
                        gentoken_prev_hyp_ids.append(hyp_id)
                        hyp_copy_info = dict()  # of (token_pos, copy_prob)
                        hyp_unk_copy_info = []

                        if args.no_copy is False:
                            # print("the no_copy is false**********************************")
                            for token, token_pos_list in aggregated_primitive_tokens.items():
                                sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0, Variable(T.LongTensor(token_pos_list))).sum()
                                gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                                if token in primitive_vocab:
                                    token_id = primitive_vocab[token]
                                    primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob

                                    hyp_copy_info[token] = (token_pos_list, gated_copy_prob.data.item())
                                else:
                                    hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                                                              'copy_prob': gated_copy_prob.data.item()})

                        if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                            # print("the hyp info is > 0 ****************************")
                            unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                            token = hyp_unk_copy_info[unk_i]['token']
                            primitive_prob[hyp_id, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]['copy_prob']
                            gentoken_new_hyp_unks.append(token)

                            hyp_copy_info[token] = (hyp_unk_copy_info[unk_i]['token_pos_list'], hyp_unk_copy_info[unk_i]['copy_prob'])

            new_hyp_scores = None
            if applyrule_new_hyp_scores:
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores))
            if gentoken_prev_hyp_ids:
                primitive_log_prob = torch.log(primitive_prob)
                gen_token_new_hyp_scores = (hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids, :]).view(-1)

                if new_hyp_scores is None: new_hyp_scores = gen_token_new_hyp_scores
                else: new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0), beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                action_info = ActionInfo()
                if new_hyp_pos < len(applyrule_new_hyp_scores):
                    # it's an ApplyRule or Reduce action
                    prev_hyp_id = applyrule_prev_hyp_ids[new_hyp_pos]
                    prev_hyp = hypotheses[prev_hyp_id]

                    prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                    # ApplyRule action
                    if prod_id < len(self.grammar):
                        production = self.grammar.id2prod[prod_id]
                        action = ApplyRuleAction(production)
                    # Reduce action
                    else:
                        action = ReduceAction()
                else:
                    # it's a GenToken action
                    token_id = (new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(1)

                    k = (new_hyp_pos - len(applyrule_new_hyp_scores)) // primitive_prob.size(1)
                    # try:
                    # copy_info = gentoken_copy_infos[k]
                    prev_hyp_id = gentoken_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id]
                    # except:
                    #     print('k=%d' % k, file=sys.stderr)
                    #     print('primitive_prob.size(1)=%d' % primitive_prob.size(1), file=sys.stderr)
                    #     print('len copy_info=%d' % len(gentoken_copy_infos), file=sys.stderr)
                    #     print('prev_hyp_id=%s' % ', '.join(str(i) for i in gentoken_prev_hyp_ids), file=sys.stderr)
                    #     print('len applyrule_new_hyp_scores=%d' % len(applyrule_new_hyp_scores), file=sys.stderr)
                    #     print('len gentoken_prev_hyp_ids=%d' % len(gentoken_prev_hyp_ids), file=sys.stderr)
                    #     print('top_new_hyp_pos=%s' % top_new_hyp_pos, file=sys.stderr)
                    #     print('applyrule_new_hyp_scores=%s' % applyrule_new_hyp_scores, file=sys.stderr)
                    #     print('new_hyp_scores=%s' % new_hyp_scores, file=sys.stderr)
                    #     print('top_new_hyp_scores=%s' % top_new_hyp_scores, file=sys.stderr)
                    #
                    #     torch.save((applyrule_new_hyp_scores, primitive_prob), 'data.bin')
                    #
                    #     # exit(-1)
                    #     raise ValueError()

                    if token_id == primitive_vocab.unk_id:
                        if gentoken_new_hyp_unks:
                            token = gentoken_new_hyp_unks[k]
                        else:
                            token = primitive_vocab.id2word[primitive_vocab.unk_id]
                    else:
                        token = primitive_vocab.id2word_[token_id.item()]

                    action = GenTokenAction(token)
                    # print("the token",token,action)
                    if token in aggregated_primitive_tokens:
                        action_info.copy_from_src = True
                        action_info.src_token_position = aggregated_primitive_tokens[token]

                    if debug:
                        action_info.gen_copy_switch = 'n/a' if args.no_copy else primitive_predictor_prob[prev_hyp_id, :].log().cpu().data.numpy()
                        action_info.in_vocab = token in primitive_vocab
                        action_info.gen_token_prob = gen_from_vocab_prob[prev_hyp_id, token_id].log().cpu().data.item() \
                            if token in primitive_vocab else 'n/a'
                        action_info.copy_token_prob = torch.gather(primitive_copy_prob[prev_hyp_id],
                                                                   0,
                                                                   Variable(T.LongTensor(action_info.src_token_position))).sum().log().cpu().data.item() \
                            if args.no_copy is False and action_info.copy_from_src else 'n/a'

                action_info.action = action
                action_info.t = t
                if t > 0:
                    action_info.parent_t = prev_hyp.frontier_node.created_time
                    action_info.frontier_prod = prev_hyp.frontier_node.production
                    action_info.frontier_field = prev_hyp.frontier_field.field

                if debug:
                    action_info.action_prob = new_hyp_score - prev_hyp.score

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                # if new_hyp.code!=None:
                #     print("new_hyp",new_hyp.code)
                new_hyp.score = new_hyp_score

                if new_hyp.completed:
                    # add length normalization
                    new_hyp.score /= (t+1)
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]))
                t += 1
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        # print(completed_hypotheses)
        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

    @classmethod
    def load(cls, model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.cuda = cuda

        parser = cls(saved_args, vocab, transition_system)

        parser.load_state_dict(saved_state)

        if cuda: parser = parser.cuda()
        parser.eval()

        return parser
