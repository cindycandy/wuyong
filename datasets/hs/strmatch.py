import nltk
def BF_match(str,pattern):
    #暴力匹配算法，逐个遍历，时间复杂度为O（n*m）
    indies = []
    n = len(str)
    m = len(pattern)
    for i in range(n - m + 1):
        index = i
        for j in range(m):
            if str[index] == pattern[j]:
                index += 1
            else:
                break
        if index - i == m:
            indies.append(i)
    if len(indies) != 0:
        return True
    return False

def getBMBC(pattern):
    # 预生成坏字符表
    BMBC = dict()
    for i in range(len(pattern) - 1):
        char = pattern[i]
        # 记录坏字符最右位置（不包括模式串最右侧字符）
        BMBC[char] = i + 1
    return BMBC

def getBMGS(pattern):
    # 预生成好后缀表
    BMGS = dict()

    # 无后缀仅根据坏字移位符规则
    BMGS[''] = 0

    for i in range(len(pattern)):

        # 好后缀
        GS = pattern[len(pattern) - i - 1:]

        for j in range(len(pattern) - i - 1):

            # 匹配部分
            NGS = pattern[j:j + i + 1]

            # 记录模式串中好后缀最靠右位置（除结尾处）
            if GS == NGS:
                BMGS[GS] = len(pattern) - j - i - 1
    return BMGS

def BM_match(str,pattern):
    m = len(pattern)
    n = len(str)
    i = 0
    j = m
    indies = []
    BMBC = getBMBC(pattern=pattern)  # 坏字符表
    BMGS = getBMGS(pattern=pattern)  # 好后缀表
    while i < n:
        while (j > 0):
            if i + j - 1 >= n:  # 当无法继续向下搜索就返回值
                if len(indies) != 0:
                    return True
                return False

            # 主串判断匹配部分
            a = str[i + j - 1:i + m]

            # 模式串判断匹配部分
            b = pattern[j - 1:]

            # 当前位匹配成功则继续匹配
            if a == b:
                j = j - 1

            # 当前位匹配失败根据规则移位
            else:
                i = i + max(BMGS.setdefault(b[1:], m), j - BMBC.setdefault(str[i + j - 1], 0))
                j = m

            # 匹配成功返回匹配位置
            if j == 0:
                indies.append(i)
                i += 1
                j = len(pattern)



result = BM_match("distance","dis")
print(result)
