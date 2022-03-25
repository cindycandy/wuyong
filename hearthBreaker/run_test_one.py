# -*- coding: utf-8 -*-

import hearthbreaker.cards as cards
import sys
import re
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

def format_count(name):
	count = globals()[name]
	print(name.replace('_', ' ') + ':', count, '/', len(CARDS), '=', '%.2f%%' % (float(count) / len(CARDS) * 100))


def get_test_result(code,ref_code):
	results = ''
	assert code.startswith('class')
	assert ref_code.startswith('class')
	name = code[len('class'):].partition('(')[0].strip()
	matchedName = ref_code[len('class'):].partition('(')[0].strip()

	if matchedName not in dir(cards):
		return 'not exist'
		
	if matchedName != name:
		code = code.replace(name,matchedName)
		name = matchedName
			
		#class定义有没有出错
	module = sys.modules[getattr(cards, name).__module__]
	raw_card = getattr(module, name)
	try:
		exec(code, module.__dict__)
	except Exception as e:
		setattr(module, name, raw_card)
		return "class error"

		#创建对象有没有出错
	try:
		new_card = getattr(module, name)
		new_card()
	except Exception as e:
		setattr(module, name, raw_card)
		return "new error"

	setattr(cards, name, getattr(module, name))
	try:
		if name not in TESTS:
			return "no test"
		else:
			TESTS[name]()
	except Exception as e:
		if '!=' in str(e): #运行结果与测试用例里的assert不一致
			return 'result error ' 
		else:
			return 'other error '
	return "OK"
if __name__ == '__main__':
	get_test_result()