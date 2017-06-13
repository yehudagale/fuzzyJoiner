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

def load_good_buckets(table_string1, table_string2, con, meta):
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
            temp_list1, x = load_bucket(words1, x)
            temp_list2, y = load_bucket(words2, y)
            word_list.append([temp_list1, temp_list2])
    return word_list, bucket_words
def find_next_bucket(table, position):
    prev_word = table[position][1]
    while position < len(table):
        if table[position][1] != prev_word:
            return position
        position += 1
    return None
def load_bucket(table, position):
    bucket = []
    prev_word = table[position][1]
    while position < len(table):
        if table[position][1] != prev_word:
            return (bucket, position)
        bucket.append(table[position][0].split())
        position += 1
    return [], None
def load_buckets(table_string, con, meta):
    table = meta.tables[table_string]
    words = con.execute(select([table]))
    word_list = []
    prev_word = ""
    curr_list = []
    bucket_words = []
    for row in words:
        if row[1] != prev_word:
            bucket_words.append(prev_word)
            prev_word = row[1]
            word_list.append(curr_list)
            curr_list = []
        curr_list.append(row[0].split())
    return word_list, bucket_words
def sort_buckets(to_sort):
    for bucket in to_sort:
        for name in bucket:
            name.sort()
        bucket.sort()
    return to_sort
def make_bucket_dict(bucket_words1, bucket_words2):
    x = 0
    y = 0
    dictionary = {}
    while x < len(bucket_words1) and y < len(bucket_words2):
        if bucket_words1[x] > bucket_words2[y]:
            y += 1
        elif bucket_words1[x] < bucket_words2[y]:
            x += 1
        else:
            dictionary[x] = y
            x += 1
    return dictionary
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
    temp = ((1 + (beta * beta)) * true_positives)
    return  temp / (temp + false_positive + ((beta * beta) * false_negative))
def get_aliases(con, meta):
    table = meta.tables['aliases']
    aliases = con.execute(select([table]))
    dictionary = {}
    for row in aliases:
        dictionary[" ".join(row[0].split())] = row[1]
    return dictionary
con, meta = connect('yehuda', 'test', 'fuzzyjoin')
bucket_list, bucket_words = load_good_buckets('wordtable1', 'wordtable2', con, meta)
matches = {}
for pair in bucket_list:
    for name1 in pair[0]:
        for name2 in pair[1]:
            if name1[0] == name2[0] and name2[-1] == name1[-1]:
                matches[" ".join(name1)] = " ".join(name2)
print fscore(get_aliases(con, meta), matches, 1)
        
    