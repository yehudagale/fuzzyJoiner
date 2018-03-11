import sqlalchemy
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

# create a set of tuples for the machine learning model where each entity name is matched
# with at least 2 other names of the same entity, and each entity is paired with 2 other
# names of a different entity.  The different pairs are chosen based on the fact that the have the 
# same name part, or are completely different.  If the name is unique then the 2 different pairs
# are just 2 random pairs.
def create_tuples_for_siamese_network():
    aliases_m = meta.tables['aliases']
    word2entities_m= meta.tables['words2entities']


#this should combine several functions and return alisases and bucket lists.
def condensed_start():
    pass
def load_good_buckets(table_string1, table_string2, dictionary, con, meta):
    table = meta.tables[table_string1]
    words1 = [list(row) for row in con.execute(select([table]))]
    table = meta.tables[table_string2]
    words2 = [list(row) for row in con.execute(select([table]))]
    buckets = []
    bucket_words = {}
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
            word = words1[x][1]
            temp_list1, x = load_bucket(words1, x, dictionary)
            temp_list2, y = load_bucket(words2, y, dictionary)
            pair = [temp_list1, temp_list2]
            buckets.append(pair)
            bucket_words[word] = buckets[-1]
    return buckets, bucket_words
def find_next_bucket(table, position):
    prev_word = table[position][1]
    while position < len(table):
        if table[position][1] != prev_word:
            return position
        position += 1
    return None
def load_bucket(table, position, dictionary):
    bucket = []
    prev_word = table[position][1]
    while position < len(table):
        if table[position][1] != prev_word:
            return (bucket, position)
        bucket.append([dictionary[table[position][0]], table[position][0]])
        position += 1
    return bucket, None
def create_double_num_dicts(aliases):
    serial_num = 0
    num_to_word = {}
    word_to_num = {}
    for pair in aliases:
        for name in pair:
            num_to_word[serial_num] = name
            word_to_num[name] = serial_num
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
            false_negative += 1
    print( "total names: " + str(len(true_items)))
    print( "not indexed: " + str(not_indexed))
    print ("wrongly indexed: " + str(false_negative - not_indexed))
    print ("all false negitives: "  + str(false_negative))
    print ("true positives: " + str(true_positives))
    print ("false positives: " + str(false_positive))
    temp = ((1 + (beta * beta)) * true_positives)
    return  temp / (temp + false_positive + ((beta * beta) * false_negative))
def test_key(true_items, test_dict, key):
    for answer in test_dict[key]:
        if (key, answer) in true_items:
            return True
    return False
def make_test_dict(items, k):
    dictionary = {}
    overflow = 0
    for pair in items:
        if pair[0] in dictionary:
            if len(dictionary[pair[0]]) < k:
                dictionary[pair[0]].append(pair[1])
            else:
                overflow += 1
        else:
            dictionary[pair[0]] = [pair[1]]
    print( "overflow: " + str(overflow))
    return dictionary
def get_aliases(con, meta):
    table = meta.tables['aliases']
    aliases = con.execute(select([table]))
    dictionary = set([])
    for row in aliases:
        dictionary.add((row[0], row[1]))
    return dictionary
def run_test(pre_procces, test, num_to_word, bucket_list):
    bucket_list = pre_proccess_words(num_to_word, bucket_list, pre_procces)
    matches = set([])
    for pair in bucket_list:
        for name1 in pair[0]:
            for name2 in pair[1]:
                if test(name1[1], name2[1]):
                    matches.add((num_to_word[name1[0]], num_to_word[name2[0]]))
    return matches
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
def export_unbucketed(impossible, con, meta):
    execute_pairs = []
    if 'unbucketed' in meta.tables:
        meta.tables['unbucketed'].drop(con)
    unbucketed = Table('unbucketed', meta, Column('name1', String), Column('name2', String), extend_existing=True)
    zipping_string = ('name1', 'name2')
    for key in impossible:
        for name in impossible[key]:
            execute_pairs.append(dict(zip(zipping_string, (key, name))))
    meta.create_all(con)
    con.execute(unbucketed.insert(), execute_pairs)
def export_missed(aliases, test_dict, con, meta):
        missed_items = get_missed(aliases, test_dict)
        execute_pairs = []
        if 'missed' in meta.tables:
            meta.tables['missed'].drop(con)
        missed = Table('missed', meta, Column('name1', String), Column('name2', String), extend_existing=True)
        zipping_string = ('name1', 'name2')
        for pair in missed_items:
            execute_pairs.append(dict(zip(zipping_string, pair)))
        meta.create_all(con)
        con.execute(missed.insert(), execute_pairs)
def get_impossible(aliases, bucket_list, num_to_word):
    testing_dict = {}
    for pair in aliases:
        if pair[0] in testing_dict:
            testing_dict[pair[0]].append(pair[1])
        else:
            testing_dict[pair[0]] = [pair[1]]
    bucket_list = pre_proccess_words(num_to_word, bucket_list, lambda x : x)
    for bucket_pair in bucket_list:
        other_bucket = set([bucket_pair[1][x][1] for x in range(len(bucket_pair[1]))])
        for name in bucket_pair[0]:
            other_name = 0
            while other_name < len(testing_dict[name[1]]):
                try:
                    if testing_dict[name[1]][other_name] in other_bucket:
                            testing_dict[name[1]].pop(other_name)
                except ValueError:
                    pass
                other_name += 1
    for key in testing_dict.copy():
        if not testing_dict[key]:
            del testing_dict[key]
    return testing_dict
def pre_proccess_words(num_to_word, bucket_list, function):
    for pair in bucket_list:
        for bucket in pair:
            for name in bucket:
                name[1] = function(num_to_word[name[0]])
    return bucket_list
def run_special_test(bucket_list, num_to_word):
    matches = set([])
    for pair in bucket_list:
        if len(pair[0]) == 1 and len(pair[1]) == 1:
            matches.add((num_to_word[pair[0][0][0]], num_to_word[pair[1][0][0]]))
    return matches
