# coding=utf-8

from __future__ import print_function

import token as tk
try:
    from cStringIO import StringIO
except:
    from io import StringIO
from tokenize import generate_tokens

# 总结：这一页只包含一个函数，也就是tokenize_code,在该函数下，主要针对str类型的进行处理
# decode模式下使其包含双引号正常加入，规范化模式下用STR代表str类
# 最后返回的是token化的结果，可直接运行本py文件观察效果

def tokenize_code(code, mode=None):
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.ENDMARKER:
            break

        if mode == 'decoder':
            if toknum == tk.STRING:
                quote = tokval[0]
                tokval = tokval[1:-1]
                tokens.append(quote)
                tokens.append(tokval)
                tokens.append(quote)    #这里是将“tokval”包括双引号加入,下面的dedent是缩进的意思，不会将其放入tokens中
            elif toknum == tk.DEDENT:
                continue
            else:
                tokens.append(tokval)
        elif mode == 'canonicalize':
            if toknum == tk.STRING:
                tokens.append('_STR_')
            elif toknum == tk.DEDENT:
                continue
            else:
                tokens.append(tokval)
        else:
            tokens.append(tokval)

    return tokens


if __name__ == '__main__':
    print(tokenize_code('torch.readfile(“a.txt”“,0)'))
