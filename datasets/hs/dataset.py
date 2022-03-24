# coding=utf-8

from __future__ import print_function

import torch
import re
import pickle
import ast
import astor
import nltk
import sys

import numpy as np

from asdl.lang.py3.py3_transition_system import python_ast_to_asdl_ast, asdl_ast_to_python_ast, Python3TransitionSystem
from asdl.hypothesis import *

from components.action_info import ActionInfo, get_action_infos
import itertools

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')

#这句话的作用是
#(?P<quote>['\"])匹配单引号或者双引号 括起来的字符串,(?P<string>.*?)匹配
QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")

# def get_link(nl, code):
#     relation = np.zeros((len(nl), len(code)), dtype=np.int64)
#     print(nl,"\n",code)
#     point_a_1 = nl.index("NAME_END")
#     point_a_2 = nl.index("RARITY_END")
#     point_a_3 = len(nl)
#     point_b_1 = 0
#     point_b_2 = code.index("API_END")
#     point_b_3 = code.index("CODEELEM_END")
#     #默认code总第一个字符就是class名字
#     for i in range(point_a_1):
#         if nl[i] in code[0]:
#             relation[i][0] = 1
#     for i in range(point_a_1+1,point_a_2):
#         for j in range(point_b_2 + 1, point_b_3):
#             q = nl[i].lower()
#             c = code[j].lower()
#             if q in c:
#                 relation[i][j] = 1
#     for i in range(point_a_2+1,point_a_3):
#         for j in range(point_b_1 + 1, point_b_2):
#             q = nl[i].lower()
#             c=code[j].lower()
#             sim_q = ""
#             if len(q) < 3:
#                 continue
#             if q == "opponent":
#                 sim_q = "enemy"
#             if q==c:
#                 relation[i,j] = 1
#                 # print(q,c)
#             elif q in c:
#                 # print("look at the link",q,c)
#                 relation[i,j] = 1
#                 # print(q,c)
#             elif sim_q!= "" and sim_q in c:
#                 # print("look at the sim link", q, c)
#                 relation[i, j] = 1
#             elif c in q:
#                 relation[i][j] = 1
#     print(relation)
#     if np.sum(relation) < 3:
#         print("Alert: the sum is less than 3")
#     return relation

def get_link(nl,code):
    print(nl,"\n",code)
    relation = np.zeros((len(nl), len(code)), dtype=np.int64)
    for i,q in enumerate(nl):
        for j,c in enumerate(code):
            q=q.lower()
            c=c.lower()
            sim_q = ""
            if q == "opponent":
                sim_q = "enemy"
            # print(q,c)
            if len(q)<3:
                continue
            if q==c:
                relation[i,j] = 1
                # print(q,c)
            elif q in c:
                # print("look at the link",q,c)
                relation[i,j] = 2
                # print(q,c)
            elif sim_q!= "" and sim_q in c:
                # print("look at the sim link", q, c)
                relation[i, j] = 3
            elif c in q:
                relation[i,j] = 4
                # print("code in ques",q,c)
    # print(relation)
    if np.sum(relation) < 3:
        print("Alert: the sum is less than 3",relation)
    # if nl[0] == "Innervate" or nl[0] == "Magma":
    #     print(nl,code,relation)
    return relation

def compute_relation(relation):
    w, h = relation.shape
    relation_len = w + h
    new_relations = np.zeros((relation_len,relation_len), dtype=np.int64)
    new_relations[0:w, w:w + h] = relation
    new_relations[w:w+h, 0:w] = relation.transpose()
    # sum_value = sum(new_relations)
    if new_relations.any() == False:
        print("!!!!!!!!!value is zero")
    return new_relations

def fixRelationWithPosition(relation,q_len):
    t_len = len(relation)
    #api对应api的关系修改后，11的值要变为最下面那个循环里的最大值
    fixed_relations = [[j+11 if j!=0 else 0 for j in i] for i in relation]
    #query对应api的默认关系为5
    fixed_relations[0:q_len,q_len:t_len] = [[5 if j==0 else j for j in i] for i in fixed_relations[0:q_len,q_len:t_len]]
    #api对应query的默认关系为6
    fixed_relations[q_len:t_len,0:q_len] = [[6 if j==0 else j for j in i] for i in fixed_relations[q_len:t_len,0:q_len]]
    #query和query的关系，之后不再修改
    for i in range(q_len):
        for j in range(q_len):
            if i==j:fixed_relations[i][j] = 2
            elif i-j==1:fixed_relations[i][j] = 1
            elif i-j >= 2:fixed_relations[i][j] = 0
            elif j-i == 1:fixed_relations[i][j] = 3
            elif j-i >= 2:fixed_relations[i][j] = 4
    #api和api的关系，这里只添加距离信息,否则加回if判断
    for i in range(q_len,t_len):
        for j in range(q_len,t_len):
            # if fixed_relations[i][j]!=0:
            if i==j:fixed_relations[i][j] = 9
            elif i-j==1:fixed_relations[i][j] = 8
            elif i-j >= 2:fixed_relations[i][j] = 7
            elif j-i == 1:fixed_relations[i][j] = 10
            elif j-i >= 2:fixed_relations[i][j] = 11
    return fixed_relations


def get_matched_api(data):
    # print(data.replace('§','\n'))
    matched = []
    gold_ast_tree = ast.parse(data).body[0]
    #由此处观察整个ast的结构
    # print(astor.dump_tree(gold_ast_tree))
    n = 0
    for node in ast.walk(gold_ast_tree):
        # print(n)
        # n =n+1
        # print(type(node))
        if type(node) is ast.ClassDef:matched.append(node.name)
        if type(node) is ast.FunctionDef:
            matched.append(node.name)
            # print("FunctionDef", node.name)
        if type(node) is ast.Call:
            # print("func",node.func)
            if type(node.func) is ast.Name:
                matched.append(node.func.id)
                # print("Call", node.func.id)
            if type(node.func) is ast.Attribute:
                matched.append(node.func.attr)
                # print("Call", node.func.id)
        if type(node) is ast.keyword:
            # print("keywords",node.arg)
            matched.append(node.arg)
            # print("value", node.value)
            # if type(node.value) is ast.Attribute:
            #     # matched.append(node.attr)
            #     print("Attribute", node.value.attr)
        # if type(node) is ast.Attribute:
        #     if type(node.value) is ast.Name:
        #         print("Name",node.value.id)
        #     tmp_node = node
        #     while type(tmp_node) is ast.Attribute:
        #         # matched.append(tmp_node.attr)
        #         print("Attribute", tmp_node.attr)
        #         tmp_node = tmp_node.value

    while '__init__' in matched:
        matched.remove('__init__')
    while 'super' in matched:
        matched.remove('super')
    matched.append('API_END')
    # print("matched",matched)
    return matched

# def get_matched_api(data):
#     # print(data.replace('§','\n'))
#     matched = []
#     gold_ast_tree = ast.parse(data).body[0]
#     # 这个方法当前不太好用，主要是由于同时存在visit_xxx时，调用该函数实例.visit只会调用第一个visit_xxx
#     class CodeVisitor(ast.NodeVisitor):
#         #两个方法不是完全对应的，上面的walk好像更完全，下面的node能访问到（）前的方法
#         def visit_ClassDef(self, node):
#             if isinstance(node,ast.ClassDef):
#                 matched.append(node.name)
#                 # print("hahha",node.name)
#         def visit_Assign(self, node):
#             # print("222",node)
#             if isinstance(node,ast.Assign):
#                 matched.append(node.name)
#                 # print("hahha",node.name)
#         def visit_Call(self, node):
#             if isinstance(node.func, ast.Attribute):
#                 matched.append(node.func.attr)
#                 # print("___attribute___", node.func.attr, node.func.value,node.func.ctx,node,node.func)
#             # if isinstance(type(node), ast.Attribute):
#             #     # matched.append(node.func.attr)
#             #     print("___taunt___", node.target, node.func.value,node.func.ctx,node.attr)
#             elif isinstance(node.func, ast.Name):
#                 matched.append(node.func.id)
#                 # print("__name__",type(node.func),node.func.id)
#             elif isinstance(node.func, ast.Call):
#                 matched.append(node.func.func.id)
#             else:
#                 print("call",type(node.func),node.func)
#             self.generic_visit(node)
#
#     visitor = CodeVisitor()
#     visitor.visit_Assign(gold_ast_tree)
#     if '__init__' in matched:
#         matched.remove('__init__')
#     while 'super' in matched:
#         matched.remove('super')
#     matched.append('API_END')
#     # print("matched",matched)
#     return matched

def get_matched_name(data):
    matched = []
    for name in ["SpellCard","MinionCard","WeaponCard","SecretCard"]:
        if name in data:
            matched.append(name)
    for name in ["CHARACTER_CLASS.ALL","CHARACTER_CLASS.MAGE",
                "CHARACTER_CLASS.HUNTER","CHARACTER_CLASS.SHAMAN","CHARACTER_CLASS.WARRIOR",
                "CHARACTER_CLASS.DRUID","CHARACTER_CLASS.PRIEST","CHARACTER_CLASS.PALADIN",
                "CHARACTER_CLASS.ROGUE","CHARACTER_CLASS.WARLOCK","CHARACTER_CLASS.LORD_JARAXXUS",
                "CHARACTER_CLASS.DREAM"]:
        if name in data:
            matched.append(name)
    for name in ["CARD_RARITY.FREE","CARD_RARITY.COMMON","CARD_RARITY.RARE",
                "CARD_RARITY.EPIC","CARD_RARITY.LEGENDARY"]:
        if name in data:
            matched.append(name)  
    for name in ["MINION_TYPE.ALL","MINION_TYPE.NONE","MINION_TYPE.BEAST",
                "MINION_TYPE.MURLOC","MINION_TYPE.DRAGON","MINION_TYPE.GIANT",
                "MINION_TYPE.PIRATE","MINION_TYPE.TOTEM","MINION_TYPE.MECH"]:
        if name in data:
            matched.append(name)       
    matched.append('CODEELEM_END')
    return matched


def replace_string_ast_nodes(py_ast, str_map):
    for node in ast.walk(py_ast):
        if isinstance(node, ast.Str):
            str_val = node.s

            if str_val in str_map:
                node.s = str_map[str_val]
            else:
                # handle cases like `\n\t` in string literals
                for key, val in str_map.items():
                    str_literal_decoded = key.decode('string_escape')
                    if str_literal_decoded == str_val:
                        node.s = val


class HS(object):
    @staticmethod
    def canonicalize_code(code):
        # if p_elif.match(code):
        #     code = 'if True: pass\n' + code

        # if p_else.match(code):
        #     code = 'if True: pass\n' + code

        # if p_try.match(code):
        #     code = code + 'pass\nexcept: pass'
        # elif p_except.match(code):
        #     code = 'try: pass\n' + code
        # elif p_finally.match(code):
        #     code = 'try: pass\n' + code

        # if p_decorator.match(code):
        #     code = code + '\ndef dummy(): pass'

        # if code[-1] == ':':
        #     code = code + 'pass'
        code = code.replace("搂", '\n')
        return code

    @staticmethod
    def canonicalize_str_nodes(py_ast, str_map):
        for node in ast.walk(py_ast):
            if isinstance(node, ast.Str):
                str_val = node.s

                if str_val in str_map:
                    node.s = str_map[str_val]
                else:
                    # handle cases like `\n\t` in string literals
                    for str_literal, slot_id in str_map.items():
                        str_literal_decoded = str_literal#.decode('string_escape')
                        if str_literal_decoded == str_val:
                            node.s = slot_id

    @staticmethod
    def canonicalize_query(query):
        """
        canonicalize the query, replace strings to a special place holder
        """
        str_count = 0
        str_map = dict()

        #在hs的自然语言中，并不存在被双引号包含的内容，因此此段跳过
        # matches = QUOTED_STRING_RE.findall(query)
        # # de-duplicate
        # cur_replaced_strs = set()
        # for match in matches:
        #     # If one or more groups are present in the pattern,
        #     # it returns a list of groups
        #     quote = match[0]
        #     #这里是被引用的东西，即为双引号的内容
        #     str_literal = match[1]
        #     quoted_str_literal = quote + str_literal + quote
        #
        #     if str_literal in cur_replaced_strs:
        #         # replace the string with new quote with slot id
        #         query = query.replace(quoted_str_literal, str_map[str_literal])
        #         continue
        #
        #     # FIXME: substitute the ' % s ' with
        #     #只有当str_literal为%s时，判断为true
        #     if str_literal in ['%s']:
        #         continue
        #
        #     str_repr = '_STR:%d_' % str_count
        #     str_map[str_literal] = str_repr
        #     #发生了nl被替代掉了name——end字符，并且大量词汇缺失，说明此处不适合hs
        #     # query = query.replace(quoted_str_literal, str_repr)
        #
        #     str_count += 1
        #     cur_replaced_strs.add(str_literal)

        # tokenize
        query_tokens = nltk.word_tokenize(query)

        new_query_tokens = []
        # break up function calls like foo.bar.func
        for token in query_tokens:
            new_query_tokens.append(token)
            i = token.find('.')
            if 0 < i < len(token) - 1:
                new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
                new_query_tokens.extend(new_tokens)

        query = ' '.join(new_query_tokens)
        query = query.replace('\' % s \'', '%s').replace('\" %s \"', '%s')

        return query, str_map

    @staticmethod
    def canonicalize_example(query, code):

        canonical_query, str_map = HS.canonicalize_query(query)
        query_tokens = canonical_query.split(' ')

        canonical_code = HS.canonicalize_code(code)
        # print(canonical_code)
        ast_tree = ast.parse(canonical_code)

        HS.canonicalize_str_nodes(ast_tree, str_map)
        canonical_code = astor.to_source(ast_tree)

        return query_tokens, canonical_code, str_map

    @staticmethod
    def parse_hs_dataset(annot_file, code_file, asdl_file_path, max_query_len=70, vocab_freq_cutoff=10,mod='origin'):
        asdl_text = open(asdl_file_path).read()
        grammar = ASDLGrammar.from_text(asdl_text)
        transition_system = Python3TransitionSystem(grammar)

        loaded_examples = []

        print("Starting to parse dataset of HeartStone")
        from components.vocab import Vocab, VocabEntry
        from components.dataset import Example

        for idx, (src_query, tgt_code) in enumerate(zip(open(annot_file), open(code_file))):
            src_query = src_query.strip()
            tgt_code = tgt_code.replace("搂",'\n').strip()

            src_query_tokens, tgt_canonical_code, str_map = HS.canonicalize_example(src_query, tgt_code)
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",src_query_tokens)
            python_ast = ast.parse(tgt_canonical_code)
            gold_source = astor.to_source(python_ast).strip()
            tgt_ast = python_ast_to_asdl_ast(python_ast, transition_system.grammar)
            tgt_actions = transition_system.get_actions(tgt_ast)

            # sanity check
            hyp = Hypothesis()
            for t, action in enumerate(tgt_actions):
                assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
                if isinstance(action, ApplyRuleAction):
                    # print(action.production)
                    # print(transition_system.get_valid_continuating_productions(hyp))
                    assert action.production in transition_system.get_valid_continuating_productions(hyp)

                p_t = -1
                f_t = None
                if hyp.frontier_node:
                    p_t = hyp.frontier_node.created_time
                    f_t = hyp.frontier_field.field.__repr__(plain=True)

                # print('\t[%d] %s, frontier field: %s, parent: %d' % (t, action, f_t, p_t))
                hyp = hyp.clone_and_apply_action(action)

            assert hyp.frontier_node is None and hyp.frontier_field is None

            src_from_hyp = astor.to_source(asdl_ast_to_python_ast(hyp.tree, grammar)).strip()
            assert src_from_hyp == gold_source

            related_code = []
            relation = None
            position_relation = None
            dataflow_relation = None
            if mod == 'origin':
                pass

            elif mod == 'hard': 
                matched = get_matched_api(tgt_code)
                matched_name = get_matched_name(tgt_code)
                related_code = matched + matched_name
                orig_relation = get_link(src_query_tokens, related_code)
                relation = compute_relation(orig_relation)
                dataflow_relation = (relation!=0).astype(int)
                position_relation = fixRelationWithPosition(relation,len(src_query_tokens))

            elif mod == "sg":
                pass
                # print("the hard mode to use relations: ", relation, new_relation)

            print('+' * 60)

            loaded_examples.append({'src_query_tokens': src_query_tokens,
                                    'tgt_canonical_code': gold_source,
                                    'tgt_ast': tgt_ast,
                                    'tgt_actions': tgt_actions,
                                    'raw_code': tgt_code,
                                    'str_map': str_map,
                                    'related_code': related_code,
                                    'relation': dataflow_relation,
                                    'fixed_relation': position_relation})

            # print('first pass, processed %d' % idx, file=sys.stderr)

        train_examples = []
        dev_examples = []
        test_examples = []

        action_len = []

        for idx, e in enumerate(loaded_examples):
            # print(e['related_code'])
            src_query_tokens = e['src_query_tokens'][:max_query_len]+e['related_code']
            tgt_actions = e['tgt_actions']
            tgt_action_infos = get_action_infos(src_query_tokens, tgt_actions)

            example = Example(idx=idx,
                              src_sent=src_query_tokens,
                              tgt_actions=tgt_action_infos,
                              tgt_code=e['tgt_canonical_code'],
                              tgt_ast=e['tgt_ast'],
                              meta={'raw_code': e['raw_code'], 'str_map': e['str_map']},
                              related_code=e['related_code'],
                              src_query=e['src_query_tokens'],
                              relation=e['relation'],
                              fixed_relation=e['fixed_relation'],)

            # print('second pass, processed %d' % idx, file=sys.stderr)

            action_len.append(len(tgt_action_infos))

            # train, valid, test split
            #xs:0,5,7,8
            #s ：0,64,96,128  重分：0,96,112,128
            #m ：0,280,306,332
            #big: 0,533,599,665
            if 0 <= idx < 533:
                train_examples.append(example)
            elif 533 <= idx < 599:
                dev_examples.append(example)
            else:
                test_examples.append(example)

        print('Max action len: %d' % max(action_len), file=sys.stderr)
        print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
        print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

        src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], size=5000, freq_cutoff=vocab_freq_cutoff)

        primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                            for e in train_examples]
        apis = [e.related_code for e in train_examples]

        primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=vocab_freq_cutoff)
        # assert '_STR:0_' in primitive_vocab

        # generate vocabulary for the code tokens!
        code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_examples]

        code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=vocab_freq_cutoff)
        # s = 0
        # for i in apis:
        #     for j in i:
        #         # if j == "AcidicSwampOoze":print("gggggggggg")
        #         if j in src_vocab.word2id.keys():
        #             print(j)
        #             s+=1
        # print(s)
        vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
        print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

        return (train_examples, dev_examples, test_examples), vocab

    @staticmethod
    def process_hs_dataset():
        vocab_freq_cutoff = 3  # TODO: found the best cutoff threshold
        path = '../../data/hs_big_1/'
        annot_file = path + 'hs.in'
        code_file = path + 'hs.out'
        mod = "hard"
        (train, dev, test), vocab = HS.parse_hs_dataset(annot_file, code_file,
                                                                '../../asdl/lang/py3/py3_asdl.simplified.txt',
                                                                vocab_freq_cutoff=vocab_freq_cutoff,mod=mod)

        pickle.dump(train, open(path + 'train_%s.bin'%mod, 'wb'))
        pickle.dump(dev, open(path + 'dev_%s.bin'%mod, 'wb'))
        pickle.dump(test, open(path + 'test_%s.bin'%mod, 'wb'))
        pickle.dump(vocab, open(path + 'vocab_%s.freq%d.bin'%(mod,vocab_freq_cutoff), 'wb'))



if __name__ == '__main__':
    HS.process_hs_dataset()
