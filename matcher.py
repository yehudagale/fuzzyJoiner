#using tutorial https://suhas.org/sqlalchemy-tutorial/
import sqlalchemy
from sys import argv
from sqlalchemy.sql import select
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
def connect(user, password, db, host='localhost', port=5432):
    '''Returns a connection and a metadata object'''
    # We connect with the help of the PostgreSQL URL
    # postgresql://federer:grandestslam@localhost:5432/tennis
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    # The return value of create_engine() is our connection object
    con = sqlalchemy.create_engine(url, client_encoding='utf8')

    # We then bind the connection to MetaData()
    meta = sqlalchemy.MetaData(bind=con, reflect=True)

    return con, meta

def load_good_buckets(table_string1, table_string2, dictionary, function, con, meta):
    table = meta.tables[table_string1]
    words1 = [list(row) for row in con.execute(select([table]))]
    table = meta.tables[table_string2]
    words2 = [list(row) for row in con.execute(select([table]))]
    word_list = []
    bucket_words = []
    temp_list1 = []
    temp_list2 = []
    x = 0
    y = 0
    while x != None and y != None:
        if words1[x][1] < words2[y][1]:
            x = find_next_bucket(words1, x)
        elif words1[x][1] > words2[y][1]:
            y = find_next_bucket(words2, y)
        else:
            bucket_words.append(words1[x][1])
            temp_list1, x = load_bucket(words1, x, dictionary, function)
            temp_list2, y = load_bucket(words2, y, dictionary, function)
            word_list.append([temp_list1, temp_list2])
    return word_list, bucket_words
def find_next_bucket(table, position):
    prev_word = table[position][1]
    while position < len(table):
        if table[position][1] != prev_word:
            return position
        position += 1
    return None
def load_bucket(table, position, dictionary, function):
    bucket = []
    prev_word = table[position][1]
    while position < len(table):
        if table[position][1] != prev_word:
            return (bucket, position)
        bucket.append([dictionary[table[position][0]], function(table[position][0])])
        position += 1
    return [], None
def create_double_alias_dicts(con, meta):
    table = meta.tables['aliases']
    aliases = con.execute(select([table]))
    serial_num = 0
    num_to_word = {}
    word_to_num = {}
    for row in aliases:
        num_to_word[serial_num] = row[0]
        word_to_num[row[0]] = serial_num
        serial_num += 1
        num_to_word[serial_num] = row[1]
        word_to_num[row[1]] = serial_num
        serial_num += 1
    return num_to_word, word_to_num
def fscore(true_items, test_dict, beta):
    true_positives = 0.0
    false_positive = 0.0
    false_negative = 0.0
    not_indexed = 0
    for key in test_dict:
        if test_key(true_items, test_dict, key):
            true_positives += 1
        else:
            false_positive += 1
    for pair in true_items:
        if pair[0] not in test_dict:
            false_negative += 1
            not_indexed += 1
        elif pair[1] not in test_dict[pair[0]]:
            false_negative +=1
    print "total names: " + str(len(true_items))
    print "not indexed: " + str(not_indexed)
    print "wrongly indexed: " + str(false_negative - not_indexed)
    print "all false negitives: "  + str(false_negative)
    print "true positives: " + str(true_positives)
    print "false positives: " + str(false_positive)
    temp = ((1 + (beta * beta)) * true_positives)
    return  temp / (temp + false_positive + ((beta * beta) * false_negative))
def test_key(true_items, test_dict, key):
    for answer in test_dict[key]:
        if (key, answer) in true_items:
            return True
    return False
def make_test_dict(items):
    dictionary = {}
    for pair in items:
        if pair[0] in dictionary:
            if len(dictionary[pair[0]]) < 3:
                dictionary[pair[0]].append(pair[1])
        else:
            dictionary[pair[0]] = [pair[1]]
    return dictionary
def get_aliases(con, meta):
    table = meta.tables['aliases']
    aliases = con.execute(select([table]))
    dictionary = set([])
    for row in aliases:
        dictionary.add((row[0], row[1]))
    return dictionary
def run_test(pre_procces, test, args):
    con, meta = connect(args[1], args[2], args[3])
    num_to_word, word_to_num = create_double_alias_dicts(con, meta)
    bucket_list, bucket_words = load_good_buckets('wordtable1', 'wordtable2', word_to_num, pre_procces, con, meta)
    matches = set([])
    for pair in bucket_list:
        for name1 in pair[0]:
            for name2 in pair[1]:
                if test(name1[1], name2[1]):
                    matches.add((num_to_word[name1[0]], num_to_word[name2[0]]))
    return get_aliases(con, meta), matches
def create_alias_dict(con, meta):
    table = meta.tables['aliases']
    aliases = con.execute(select([table]))
    dictionary = {}
    for row in aliases:
        dictionary[row[0]] = row[1]
    return dictionary
def get_missed(aliases, test_dict):
    missed = set([])
    for pair in aliases:
        if pair[0] not in test_dict or pair[1] not in test_dict[pair[0]]:
            missed.add(pair)
    return missed
def export_missed(aliases, test_dict, con, meta):
        missed_items = get_missed(aliases, test_dict)
        execute_pairs = []
        missed = Table('missed', meta, Column('name1', String), Column('name2', String))
        zipping_string = ('name1', 'name2')
        for pair in missed_items:
            execute_pairs.append(dict(zip(zipping_string, pair)))
        meta.create_all(con)
        con.execute(missed.insert(), execute_pairs)
def get_possible(args):
    con, meta = connect(args[1], args[2], args[3])
    num_to_word, word_to_num = create_double_alias_dicts(con, meta)
    bucket_list, bucket_words = load_good_buckets('wordtable1', 'wordtable2', word_to_num, lambda x : x, con, meta)
    aliase_dict = create_alias_dict(con, meta)
    testing_alias_dict = aliase_dict.copy()
    original = len(aliase_dict)
    for bucket_pair in bucket_list:
        other_bucket = [bucket_pair[1][x][1] for x in range(len(bucket_pair[1]))]
        for name in bucket_pair[0]:
            try:
                if aliase_dict[name[1]] in other_bucket:
                        del aliase_dict[name[1]]
            except KeyError:
                pass
    return original - len(aliase_dict)
con, meta = connect(argv[1], argv[2], argv[3])
aliases, matches = run_test(lambda x : x.replace(" ", ""), lambda name1, name2 : name1 in name2 or name2 in name1, argv)
print "possible matches: " + str(get_possible(argv))
aliases, matches2 = run_test(lambda x : set(x.split()), lambda name1, name2 : name1.issubset(name2) or name2.issubset(name1), argv)
test_dict = make_test_dict(matches.union(matches2))
print "fscore: " + str(fscore(aliases, test_dict, 1))
export_missed(aliases, test_dict, con, meta)    
