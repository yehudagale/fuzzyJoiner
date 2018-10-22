from matcher_functions import load_good_buckets, create_double_num_dicts, connect, get_aliases
class matcher(object):
    def __init__(self, user, password, database, test_pairs, bucket_number):
        con, meta = connect(user, password, database)
        num_to_word, word_to_num = create_double_num_dicts(get_aliases(con, meta))
        bucket_list, bucket_words = load_good_buckets('wordtable1', 'wordtable2', word_to_num, con, meta)
        self.rarity_match = {}
        for pair in bucket_list:
            if len(pair[0]) <= bucket_number and len(pair[1]) <= bucket_number:
                for i in range(len(pair[0])):
                    for j in range(len(pair[1])):
                        self.rarity_match[pair[0][i][1]] = pair[1][j][1]
        self.test_pairs = test_pairs
    def special_test(self, name1, name2):
        temp1 = False
        temp2 = False
        if name1 in self.rarity_match:
            temp1 = self.rarity_match[name1] == name2
        if name2 in self.rarity_match:
            temp2 = self.rarity_match[name2] == name1
        return temp1 or temp2 
    def match(self, name1, name2):
        for test_pair in self.test_pairs:
            temp1 = test_pair[0](name1)
            temp2 = test_pair[0](name2)
            if test_pair[1](temp1, temp2):
                return True
        return self.special_test(name1, name2)
