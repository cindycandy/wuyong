import csv

from components.evaluator import Evaluator
from common.registerable import Registrable
from components.dataset import Dataset
from .util import decanonicalize_code
from .hs_eval import tokenize_for_bleu_eval,tokenize_code,de_canonicalize_code
# from .bleu_score import compute_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction,corpus_bleu
import numpy as np
import ast
import astor


# @Registrable.register('default_evaluator')
@Registrable.register('hs_evaluator')
class HSEvaluator(Evaluator):
    def __init__(self, transition_system=None, args=None):
        super(HSEvaluator, self).__init__()
        self.transition_system = transition_system
        self.default_metric = 'corpus_bleu'

    def is_hyp_correct(self, example, hyp):
        ref_code = example.tgt_code
        ref_py_ast = ast.parse(ref_code)
        ref_reformatted_code = astor.to_source(ref_py_ast).strip()

        ref_code_tokens = self.transition_system.tokenize_code(ref_reformatted_code)
        hyp_code_tokens = self.transition_system.tokenize_code(hyp.code)

        return ref_code_tokens == hyp_code_tokens

    def get_sentence_bleu(self, example, hyp):
        return sentence_bleu([tokenize_for_bleu_eval(example.tgt_code)],
                             tokenize_for_bleu_eval(hyp.decanonical_code),
                             smoothing_function=SmoothingFunction().method3)

    def evaluate_dataset(self, dataset, decode_results, fast_mode=False, args=None,verbose=True,useTest = True):
        # tokenize_code = tokenize_for_bleu_eval
        import ast
        print ('evaluate_decode_results:: ',len(dataset), len(decode_results))
        # input('>>>')
        assert len(dataset) == len(decode_results)
        
        f = f_decode = None
        if verbose:
            f = open('result/%s/hs.exact_match'%args.mod, 'w')
            exact_match_ids = []
            f_decode = open('result/%s/hs.decode_results.txt'%args.mod, 'w')
            eid_to_annot = dict()

            f_bleu_eval_ref = open('result/%s/hs.ref'%args.mod, 'w')
            f_bleu_eval_hyp = open('result/%s/hs.hyp'%args.mod, 'w')
            f_generated_code = open('result/%s/hs.geneated_code'%args.mod, 'w')

        cum_oracle_bleu = 0.0
        cum_oracle_acc = 0.0
        cum_bleu = 0.0
        cum_acc = 0.0
        cum_test_result = {}
        cum_test_score = 0 
        sm = SmoothingFunction()

        all_references = []
        all_predictions = []

        if all(len(cand) == 0 for cand in decode_results):
            print('Empty decoding results for the current dataset!')
            # return -1, -1
            return {'corpus_bleu': -1,
                    'oracle_bleu': -1,
                    'avg_sent_bleu': -1,
                    'accuracy': -1,
                    'oracle_accuracy': -1}
        examples = dataset.examples if isinstance(dataset, Dataset) else dataset
        for eid in range(len(dataset)):
            example = examples[eid]
            ref_code = example.tgt_code
            # ref_ast_tree = ast.parse(ref_code).body[0]
            # refer_source = astor.to_source(ref_ast_tree).strip()
            refer_source = ref_code
            refer_tokens = tokenize_code(refer_source)
            cur_example_correct = False

            decode_cands = decode_results[eid]
            #这里是不是该增加更多的考虑内容
            if len(decode_cands) == 0:
                # print(eid,"Empty decode cands")
                continue

            decode_cand = decode_cands[0]

            # cid, cand, ast_tree, code = decode_cand
            # ast_tree = decode_cand.tree
            code = decode_cand.code
            # code = astor.to_source(ast_tree).strip()

            # simple_url_2_re = re.compile('_STR:0_', re.))
            try:
                predict_tokens = tokenize_code(code)
            except:
                print('error in tokenizing [%s]', code)
                continue

            if refer_tokens == predict_tokens:
                cum_acc += 1
                cur_example_correct = True

                if verbose:
                    exact_match_ids.append(example.idx)
                    f.write('-' * 60 + '\n')
                    f.write('example_id: %d\n' % example.idx)
                    f.write(code + '\n')
                    f.write('-' * 60 + '\n')

            ref_code_for_bleu = ref_code
            pred_code_for_bleu = code

            # we apply Ling Wang's trick when evaluating BLEU scores
            refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
            pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)
            
            # 运行测试用例
            import os
            print("useTest",useTest)
            if useTest:
                path = '../../hearthbreaker-master/'
                from hearthBreaker.run_test_one_ import get_test_result
                test_result = get_test_result(code,ref_code)
                print("test result：",test_result)
                if test_result == "OK":
                    cum_test_score+=1
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # cmd = "python3 ../../hearthbreaker-master/run_test_one_.py -code '%s' -ref_code '%s' -path '%s'"%(code,ref_code,path)
                # print(cmd)
                # os.system(cmd)
                # with open(path+'/test_result.txt','r') as f_test_result:
                # if cum_test_result[eid] == "":
                cum_test_result[eid] = test_result
            # print cum_test_result
            # raw_input()
            #
            # The if-chunk below is for debugging purpose, sometimes the reference cannot match with the prediction
            # because of inconsistent quotes (e.g., single quotes in reference, double quotes in prediction).
            # However most of these cases are solved by cannonicalizing the reference code using astor (parse the reference
            # into AST, and regenerate the code. Use this regenerated one as the reference)

            #如果预测的结果和gold不一致，但是分词后又一模一样，就是奇怪的体现。（可以查看数据集）
            #如果acc值是1，下一个
            weired = False
            if refer_tokens_for_bleu == pred_tokens_for_bleu and refer_tokens != predict_tokens:
                # cum_acc += 1
                weired = True
            elif refer_tokens == predict_tokens:
                # weired!
                # weired = True
                pass

            shorter = len(pred_tokens_for_bleu) < len(refer_tokens_for_bleu)
            #用于计算corpus_bleu
            all_references.append([refer_tokens_for_bleu])
            all_predictions.append(pred_tokens_for_bleu)

            # try:
            ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
            #第一个计算
            bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights, smoothing_function=sm.method3)
            #cum:联合   cum_bleu：总和bleu
            cum_bleu += bleu_score
            # except:
            #    pass

            if verbose:
                print('raw_id: %d, bleu_score: %f' % (example.idx, bleu_score))

                f_decode.write('-' * 60 + '\n')
                f_decode.write('example_id: %d\n' % example.idx)
                f_decode.write('intent: \n')


                f_decode.write(' '.join(example.src_sent) + '\n')

                f_bleu_eval_ref.write(' '.join(refer_tokens_for_bleu) + '\n')
                f_bleu_eval_hyp.write(' '.join(pred_tokens_for_bleu) + '\n')

                f_decode.write('canonicalized reference: \n')
                f_decode.write(refer_source + '\n')
                f_decode.write('canonicalized prediction: \n')
                f_decode.write(code + '\n')
                f_decode.write('reference code for bleu calculation: \n')
                f_decode.write(ref_code_for_bleu + '\n')
                f_decode.write('predicted code for bleu calculation: \n')
                f_decode.write(pred_code_for_bleu + '\n')
                f_decode.write('pred_shorter_than_ref: %s\n' % shorter)
                f_decode.write('weired: %s\n' % weired)
                f_decode.write('-' * 60 + '\n')

                # for Hiro's evaluation
                f_generated_code.write(pred_code_for_bleu.replace('\n', '#NEWLINE#') + '\n')


            # compute oracle
            #oraclebleu
            best_score = 0.
            cur_oracle_acc = 0.
            for decode_cand in decode_cands[:args.beam_size]:
                # cid, cand, ast_tree, code = decode_cand
                code = decode_cand.code
                try:
                    # code = astor.to_source(ast_tree).strip()
                    predict_tokens = tokenize_code(code)

                    if predict_tokens == refer_tokens:
                        cur_oracle_acc = 1

                    
                    pred_code_for_bleu = code

                    # we apply Ling Wang's trick when evaluating BLEU scores
                    pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

                    ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
                    bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu,
                                            weights=ngram_weights,
                                            smoothing_function=sm.method3)

                    if bleu_score > best_score:
                        best_score = bleu_score

                except:
                    continue

            cum_oracle_bleu += best_score
            cum_oracle_acc += cur_oracle_acc

        cum_bleu /= len(dataset)
        cum_acc /= len(dataset)
        cum_oracle_bleu /= len(dataset)
        cum_oracle_acc /= len(dataset)
        corpus_bleuValue = corpus_bleu(all_references, all_predictions, smoothing_function=sm.method3)
        print('corpus level bleu: %f', corpus_bleuValue)
        print('sentence level bleu: %f', cum_bleu)
        print('accuracy: %f', cum_acc)
        print('oracle bleu: %f', cum_oracle_bleu)
        print('oracle accuracy: %f', cum_oracle_acc)
        print('test score: %f', cum_test_score)
 

        if verbose:
            f.write(', '.join(str(i) for i in exact_match_ids))
            f.close()
            f_decode.close()

            f_bleu_eval_ref.close()
            f_bleu_eval_hyp.close()
            f_generated_code.close()

        # return cum_bleu, cum_acc
        return {'corpus_bleu': corpus_bleuValue,
                    'oracle_bleu': cum_oracle_bleu,
                    'avg_sent_bleu': cum_bleu,
                    'accuracy': cum_acc,
                    'oracle_accuracy': cum_oracle_acc,
                    'test_score': cum_test_score}
    # def evaluate_dataset(self, dataset, decode_results, fast_mode=False, args=None):
    #     output_plaintext_file = None
    #     if args and args.save_decode_to:
    #         output_plaintext_file = open(args.save_decode_to + '.txt', 'w', encoding='utf-8')
    #     examples = dataset.examples if isinstance(dataset, Dataset) else dataset
    #     assert len(examples) == len(decode_results)

    #     # speed up, cache tokenization results
    #     if not hasattr(examples[0], 'reference_code_tokens'):
    #         for example in examples:
    #             setattr(example, 'reference_code_tokens', tokenize_for_bleu_eval(example.tgt_code))

    #     if not hasattr(decode_results[0][0], 'decanonical_code_tokens'):
    #         for i, example in enumerate(examples):
    #             hyp_list = decode_results[i]
    #             # here we prune any hypothesis that throws an error when converting back to the decanonical code!
    #             # This modifies the decode_results in-place!
    #             filtered_hyp_list = []
    #             for hyp in hyp_list:
    #                 if not hasattr(hyp, 'decanonical_code'):
    #                     try:
    #                         hyp.decanonical_code = decanonicalize_code(hyp.code, slot_map=example.meta['slot_map'])
    #                         if hyp.decanonical_code:
    #                             hyp.decanonical_code_tokens = tokenize_for_bleu_eval(hyp.decanonical_code)
    #                             filtered_hyp_list.append(hyp)
    #                     except: pass

    #             decode_results[i] = filtered_hyp_list

    #     if fast_mode:
    #         references = [e.reference_code_tokens for e in examples]
    #         hypotheses = [hyp_list[0].decanonical_code_tokens if hyp_list else [] for hyp_list in decode_results]

    #         bleu_tup = compute_bleu([[x] for x in references], hypotheses, smooth=False)
    #         bleu = bleu_tup[0]

    #         return bleu
    #     else:
    #         tokenized_ref_snippets = []
    #         hyp_code_tokens = []
    #         best_hyp_code_tokens = []
    #         sm_func = SmoothingFunction().method3
    #         sent_bleu_scores = []
    #         oracle_bleu_scores = []
    #         oracle_exact_match = []
    #         for example, hyp_list in zip(examples, decode_results):
    #             tokenized_ref_snippets.append(example.reference_code_tokens)
    #             example_hyp_bleu_scores = []
    #             if hyp_list:
    #                 for i, hyp in enumerate(hyp_list):
    #                     hyp.bleu_score = sentence_bleu([example.reference_code_tokens],
    #                                                    hyp.decanonical_code_tokens,
    #                                                    smoothing_function=sm_func)
    #                     hyp.is_correct = self.is_hyp_correct(example, hyp)

    #                     example_hyp_bleu_scores.append(hyp.bleu_score)

    #                 top_decanonical_code_tokens = hyp_list[0].decanonical_code_tokens
    #                 sent_bleu_score = hyp_list[0].bleu_score
    #                 best_hyp_idx = np.argmax(example_hyp_bleu_scores)
    #                 oracle_sent_bleu = example_hyp_bleu_scores[best_hyp_idx]
    #                 _best_hyp_code_tokens = hyp_list[best_hyp_idx].decanonical_code_tokens
    #             else:
    #                 top_decanonical_code_tokens = []
    #                 sent_bleu_score = 0.
    #                 oracle_sent_bleu = 0.
    #                 _best_hyp_code_tokens = []

    #             # write results to file
    #             if output_plaintext_file:
    #                 output_plaintext_file.write(" ".join(top_decanonical_code_tokens) + '\n')
    #             oracle_exact_match.append(any(hyp.is_correct for hyp in hyp_list))
    #             hyp_code_tokens.append(top_decanonical_code_tokens)
    #             sent_bleu_scores.append(sent_bleu_score)
    #             oracle_bleu_scores.append(oracle_sent_bleu)
    #             best_hyp_code_tokens.append(_best_hyp_code_tokens)

    #         bleu_tup = compute_bleu([[x] for x in tokenized_ref_snippets], hyp_code_tokens, smooth=False)
    #         corpus_bleu = bleu_tup[0]

    #         bleu_tup = compute_bleu([[x] for x in tokenized_ref_snippets], best_hyp_code_tokens, smooth=False)
    #         oracle_corpus_bleu = bleu_tup[0]

    #         avg_sent_bleu = np.average(sent_bleu_scores)
    #         oracle_avg_sent_bleu = np.average(oracle_bleu_scores)
    #         exact = sum([1 if h == r else 0 for h, r in zip(hyp_code_tokens, tokenized_ref_snippets)]) / float(
    #             len(examples))
    #         oracle_exact_match = np.average(oracle_exact_match)

            # return {'corpus_bleu': corpus_bleu,
            #         'oracle_corpus_bleu': oracle_corpus_bleu,
            #         'avg_sent_bleu': avg_sent_bleu,
            #         'oracle_avg_sent_bleu': oracle_avg_sent_bleu,
            #         'exact_match': exact,
            #         'oracle_exact_match': oracle_exact_match}
