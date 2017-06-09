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
con, meta = connect('yehuda', 'qetuopiyrw', 'fuzzyjoin')
word_list1, bucket_words1 = load_buckets('wordtable1', con, meta)
word_list2, bucket_words2 = load_buckets('wordtable2', con, meta)
word_list1 = sort_buckets(word_list1)
word_list2 = sort_buckets(word_list2)
bucket_dict = make_bucket_dict(bucket_words1, bucket_words2)
index_1 = 0
index_2 = 0
