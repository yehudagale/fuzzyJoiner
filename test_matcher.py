#using tutorial https://suhas.org/sqlalchemy-tutorial/
import sqlalchemy
from sqlalchemy.sql import select
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
def fscore(true_dict, test_dict, beta):
    true_positives = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for key in test_dict:
        if true_dict[key] == test_dict[key]:
            true_positives += 1
        else:
            false_positive += 1
    for key in true_dict:
        if key not in test_dict:
            false_negative +=1
        elif true_dict[key] != test_dict[key]:
            false_negative += 1
    print true_positives
    print false_positive
    print false_negative
    temp = ((1 + (beta * beta)) * true_positives)
    return  temp / (temp + false_positive + ((beta * beta) * false_negative))
def get_aliases(con, meta):
    table = meta.tables['aliases']
    aliases = con.execute(select([table]))
    dictionary = {}
    for row in aliases:
        dictionary[row[0]] = row[1]
    return dictionary
def run_test(function, test):
    con, meta = connect('yehuda', 'test', 'fuzzyjoin')
    num_to_word, word_to_num = create_double_alias_dicts(con, meta)
    bucket_list, bucket_words = load_good_buckets('wordtable1', 'wordtable2', word_to_num, function, con, meta)
    matches = {}
    for pair in bucket_list:
        for name1 in pair[0]:
            for name2 in pair[1]:
                if test(name1, name2):
                    matches[num_to_word[name1[0]]] = num_to_word[name2[0]]
    return get_aliases(con, meta), matches
aliases, matches = run_test(lambda x : set(x.split()), lambda name1, name2 : name1[1].issubset(name2[1]) or name2[1].issubset(name1[1]))
print fscore(aliases, matches, 1)
