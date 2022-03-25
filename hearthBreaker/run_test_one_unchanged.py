# -*- coding: utf-8 -*-
import argparse
import hearthbreaker.cards as cards
import sys
import re



def get_test_result(code,ref_code):
	results = ''
	assert code.startswith('class')
	assert ref_code.startswith('class')
	name = code[len('class'):].partition('(')[0].strip()
	matchedName = ref_code[len('class'):].partition('(')[0].strip()

	#dir返回参数的属性、方法列表
	if matchedName not in dir(cards):
		return 'not exist'

	if matchedName != name:
		code = code.replace(name,matchedName)
		name = matchedName

	#sys.module是一个字典，import后导入(加载)的所有模块名和模块对象；getatter可以得到对象的属性值，如果不存在会报错，可以增加第三个值表示不存在则使用该默认值
	#__module__表示当前操作的对象在哪个模块,比如当前输出是hearthbreaker.cards.minions.neutral
	module = sys.modules[getattr(cards, name).__module__]
	raw_card_name = getattr(module, name)

	#运行自己生成的代码，属性值使用neutral中的属性，__dict__是类或对象的所有属性
	# 如果异常，对模型中该名字改为框架中该方法的名字
	try:
		# print(code)
		exec(code, module.__dict__)
	except Exception as e:
		setattr(module, name, raw_card_name)
		return "class error"


	#取出该方法并运行，如果还报错，那就再改一次名字（和上面一样），并报错
	try:
		new_card = getattr(module, name)
		new_card()
	except Exception as e:
		setattr(module, name, raw_card_name)
		return "new error"

	#对cards这一模块内的名字修改
	setattr(cards, name, getattr(module, name))

	from tests.card_tests.druid_tests import TestDruid
	from tests.card_tests.hunter_tests import TestHunter
	from tests.card_tests.mage_tests import TestMage
	from tests.card_tests.neutral_tests import TestCommon as TestNeutral
	from tests.card_tests.paladin_tests import TestPaladin
	from tests.card_tests.priest_tests import TestPriest
	from tests.card_tests.rogue_tests import TestRogue
	from tests.card_tests.shaman_tests import TestShaman
	from tests.card_tests.warlock_tests import TestWarlock
	from tests.card_tests.warrior_tests import TestWarrior

	TESTS = {}
	def collect_tests(TestCase):
		testcase = TestCase()
		testcase.setUp()
		for name in dir(testcase):
			if not name.startswith('test_'):
				continue
			test = getattr(testcase, name)
			if not callable(test):
				continue
			TESTS[name[len('test_'):]] = test

	collect_tests(TestDruid)
	collect_tests(TestHunter)
	collect_tests(TestMage)
	collect_tests(TestNeutral)
	collect_tests(TestPaladin)
	collect_tests(TestPriest)
	collect_tests(TestRogue)
	collect_tests(TestShaman)
	collect_tests(TestWarlock)
	collect_tests(TestWarrior)
	try:
		if name not in TESTS:
			return "no test"
		else:
			# print('testing.....',TESTS[name])
			TESTS[name]()
	except Exception as e:
		if '!=' in str(e): #运行结果与测试用例里的assert不一致
			return 'result error '
		else:
			return 'other error '
	return "OK"

parser = argparse.ArgumentParser()
parser.add_argument('-code', default='class A')
parser.add_argument('-ref_code', default="class B")
parser.add_argument('-path', default="")

if __name__ == '__main__':
	codes_path = "./data/hs.ref"
	ref_codes_path = "./data/hs.ref"
	# data_path = "./data/hs.test_data.decode_results_02.txt"
	data_path = "data/hs.decode_results_ori_att_1212.txt"
	# with open(codes_path) as f:
	# 	codes = f.read().replace(" ","").split("\n")
	# with open(ref_codes_path) as f:
	# 	ref_codes = f.read().replace(" ","").split("\n")
	codes = []
	ref_codes = []
	start,end,mid = -1,-1,-1
	with open(data_path,"r",encoding='UTF-8') as f:data = f.readlines()
	for i, line in enumerate(data):
		if line.startswith("canonicalized reference"):
			start = i
		if line.startswith("canonicalized prediction:"):
			mid = i
		if line.startswith("reference code for bleu calculation:"):
			end = i
		if start != -1 and mid != -1:
			ref_code = data[start + 1: mid]
			ref_code = "".join(ref_code)
			ref_codes.append(ref_code)
			start = -1
		elif mid != -1 and end != -1:
			code = data[mid + 1: end]
			code = "".join(code)
			codes.append(code)
			mid,end = -1,-1
	assert len(codes) == len(ref_codes)
	t = len(codes)
	percent_pass = 0
	args = parser.parse_args()
	# code = ""
	for i in range(t):
		m = get_test_result(codes[i],ref_codes[i])
		if m == "no test":print(codes[i])
		if m == "OK":percent_pass+=1
	# percent_pass/=t
	print(percent_pass)




	a =	get_test_result('''class DarkscaleHealer(MinionCard):

    def __init__(self):
        super().__init__('Darkscale Healer', 5, CHARACTER_CLASS.ALL,
            CARD_RARITY.COMMON, battlecry=Battlecry(Heal(2),
            CharacterSelector()))

    def create_minion(self, player):
        return Minion(4, 5)''','''class DarkscaleHealer(MinionCard):

    def __init__(self):
        super().__init__('Darkscale Healer', 5, CHARACTER_CLASS.ALL,
            CARD_RARITY.COMMON, battlecry=Battlecry(Heal(2),
            CharacterSelector()))

    def create_minion(self, player):
        return Minion(4, 5)''')
	# with open(args.path+ '/test_result.txt', 'w') as f_test:
		# f_test.write(a)
	print(a)

	# python run_test_one_.py -code "class DarkscaleHealer(MinionCard):
	#
    # def __init__(self):
    #     super().__init__('Darkscale Healer', 5, CHARACTER_CLASS.ALL,
    #         CARD_RARITY.COMMON, battlecry=Battlecry(Damage(2),
    #         CharacterSelector(players=BothPlayer(), picker=UserPicker())))
	#
    # def create_minion(self, player):
    #     return Minion(4, 5)" -ref_code "class DarkscaleHealer(MinionCard):
	#
    # def __init__(self):
    #     super().__init__('Darkscale Healer', 5, CHARACTER_CLASS.ALL,
    #         CARD_RARITY.COMMON, battlecry=Battlecry(Heal(2),
    #         CharacterSelector()))
	#
    # def create_minion(self, player):
    #     return Minion(4, 5)" -path "./"