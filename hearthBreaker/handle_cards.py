import os
save_path = "result.txt"
druid_test_path = "tests/card_tests/druid_tests.py"
druid_minions_path = "hearthbreaker/cards/minions/druid.py"
druid_spell_path = "hearthbreaker/cards/spells/druid.py"
druid_weapon_path = "hearthbreaker/cards/weapons/druid.py"
druid_minions_cards = []
druid_spell_cards = []
druid_weapon_cards = []
druid_test_cards = []

heroes = ["druid","hunter","mage","neutral","paladin","priest","rogue","shaman","warlock","warrior"]
for hero in heroes:
    druid_test_path = "tests/card_tests/"+hero+"_tests.py"
    druid_minions_path = "hearthbreaker/cards/minions/"+hero+".py"
    druid_spell_path = "hearthbreaker/cards/spells/"+hero+".py"
    druid_weapon_path = "hearthbreaker/cards/weapons/"+hero+".py"

    druid_minions_cards = []
    druid_spell_cards = []
    druid_weapon_cards = []
    druid_test_cards = []

    if os.path.exists(druid_minions_path):
        for line in open(druid_minions_path,'r',encoding='utf-8'):
            if 'class ' in line:
                s = line.index(" ")
                e = line.index("(")
                card = line[s+1:e]
                druid_minions_cards.append(card)

    if os.path.exists(druid_spell_path):
        for line in open(druid_spell_path,'r',encoding='utf-8'):
            if 'class ' in line:
                s = line.index("s ")
                e = line.index("(")
                card = line[s+2:e]
                druid_spell_cards.append(card)

    if os.path.exists(druid_weapon_path):
        for line in open(druid_weapon_path,'r',encoding='utf-8'):
            if 'class ' in line:
                s = line.index("s ")
                e = line.index("(")
                card = line[s+1:e]
                druid_weapon_cards.append(card)

    if os.path.exists(druid_test_path):
        for line in open(druid_test_path,'r',encoding='utf-8'):
            if 'def test_' in line:
                s = line.index("_")
                e = line.index("(")
                card = line[s+1:e]
                druid_test_cards.append(card)

    # print(druid_minions_cards)
    # print(druid_spell_cards)
    # print(druid_weapon_cards)
    # print(druid_test_cards)
    #
    test_card_unknown = []
    minion_card_tested = []
    spell_card_tested = []
    weapon_card_tested = []
    for card in druid_test_cards:
        if card in druid_minions_cards:
            minion_card_tested.append(card)
        elif card in druid_spell_cards:
            spell_card_tested.append(card)
        elif card in druid_weapon_cards:
            weapon_card_tested.append(card)
        else:
            test_card_unknown.append(card)

    minion_card_untested = [card for card in druid_minions_cards if card not in minion_card_tested]
    spell_card_untested = [card for card in druid_spell_cards if card not in spell_card_tested]
    weapon_card_untested = [card for card in druid_weapon_cards if card not in weapon_card_tested]

    hero_str = "\n\n\n!!!" + hero + "\n"
    str1 = "\n!!minion_card_untested\n"+" ".join(minion_card_untested)
    str2 = "\n!!spell_card_untested\n"+" ".join(spell_card_untested)
    str3 = "\n!!weapon_card_untested\n"+" ".join(weapon_card_untested)
    str4 = "\n!!unknown_card_tested\n"+" ".join(test_card_unknown)
    if len(weapon_card_untested) == 0:
        str3 = "\n!!weapon_card_untested\n" + "null"

    with open(save_path,'a+',encoding='utf-8') as f:
        f.write(hero_str)
        f.write(str1)
        f.write(str2)
        f.write(str3)
        f.write(str4)
    # print(weapon_card_untested)
