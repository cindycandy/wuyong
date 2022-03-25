# coding=utf-8
from __future__ import print_function

import sys
import traceback
from tqdm import tqdm


def decode(examples, model, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    is_wikisql = args.parser == 'wikisql_parser'

    decode_results = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        #1111
        # unchanged = True
        # if is_wikisql:
        #     hyps = model.parse(example.src_sent, example.relation, context=example.table, beam_size=args.beam_size)
        # else:
        #     #用新的attention
        #     if unchanged==False:
        #         hyps = model.parse(example.src_query, [example.relation], example.related_code, context=None, beam_size=args.beam_size,unchanged=unchanged)
        #     else:
        if args.att_mode == "rat":
            hyps = model.parse(example.src_sent,relation=[example.fixed_relation], context=None, beam_size=args.beam_size,unchanged=True)
        if args.att_mode == "plus":
            hyps = model.parse(example.src_sent,relation=[example.relation], context=None, beam_size=args.beam_size,unchanged=True)

        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                got_code = True
                # print(hyp_id, "we have got code", hyp.code)
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
                                                                                             ' '.join(example.src_sent),
                                                                                             example.tgt_code,
                                                                                             hyp_id,
                                                                                             hyp.tree.to_string()), file=sys.stdout)
                    if got_code:
                        traceback.print_exc(file=sys.stdout)
                        print('-' * 60, file=sys.stdout)

        count += 1

        decode_results.append(decoded_hyps)

    if was_training: model.train()

    return decode_results


def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
    # print("the evaluation is begining",examples[0].relation)
    decode_results = decode(examples, parser, args, verbose=verbose)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only, args=args,useTest = True)

    # print("*"*20)
    # print("the result:",decode_results)
    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
