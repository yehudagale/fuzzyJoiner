from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from matcher_functions import connect, get_aliases, load_good_buckets, create_double_num_dicts 
from matcher_class import matcher
import os
from sqlalchemy import Table, Column, Integer, String, ForeignKey, Float
import sys
from sys import argv
import numpy as np

import random

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Flatten, Dropout, Lambda

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model, model_from_json, Sequential

from keras.utils import to_categorical

from keras.optimizers import RMSprop

from keras import backend as K

BASE_DIR = './Machine_Learning/'

# directory containing glove encodings from Wikipedia (we can swap this out for another encoding later)

# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/

GLOVE_DIR = BASE_DIR + 'glove/'

TEXT_DATA_DIR = BASE_DIR + 'nerData'

# number of words an entity is allowed to have

# distribution of number of words in peoples names can be found in peopleNamesDisbn

# distribution of number of words in company names can be found in companyNamesDisbn

# Note most of the names above that are fairly esoteric or just plain noise.  Included is

# python code to remove them

MAX_SEQUENCE_LENGTH = 10

def get_no_match_texts(argv, texts1):
    def get_non_match(name1, bucket_words, matching_set):
        for word in name1.split(" "):
            if word in bucket_words:
                bucket = bucket_words[word]
            else:
                print(word)
                return None
            if len(bucket[1]) > 1:
                for name2 in bucket[1]:
                    if (name1, name2[1]) not in matching_set:
                        return name2[1]
        return None
    no_match_texts = []
    #this should not be done here and needs to be fixed up before more work is done
    #it should instead be done by a singel function in matcher_functions
    #establish connection to database
    con, meta = connect(argv[1], argv[2], argv[3])
    #load pairs from database
    aliases = get_aliases(con, meta)
    #create dictionaries assingning serial numbers to names and names from serial numbers
    num_to_word, word_to_num = create_double_num_dicts(aliases)
    #load the buckets from the database bucket_list is aranges as follows:
    #bucket_list[pair_of_buckets][bucket(this must be 0 or 1)][name (this represents a single name)][0 for number and 1 for pre-procced name]
    bucket_list, bucket_words = load_good_buckets('wordtable1', 'wordtable2', word_to_num, con, meta)
    for index in range(len(texts1)):
        new_text = get_non_match(texts1[index], bucket_words, aliases)
        if new_text == None:
            new_text = texts1[(index + 1) % len(texts1)]
        no_match_texts.append(new_text)
    return no_match_texts

texts1 = []  # list of text samples in part 1
texts2 = [] # list of text samples in part 2

labels_index = {}  # dictionary mapping label name to numeric id - here the label name is just the name of the file in the data dir



name_list = sorted(os.listdir(TEXT_DATA_DIR))
name = name_list[0]
label_id = len(labels_index)

labels_index[name] = label_id

fpath = os.path.join(TEXT_DATA_DIR, name)

if os.path.isdir(fpath):

    raise ValueError('bad data directory')

if sys.version_info < (3,):

    f = open(fpath)

else:

    f = open(fpath, encoding='latin-1')


#change to get from sql and not read from file
for t in f.readlines():

    num_tokens = len(t.strip().split(' '))

    if 0 < num_tokens < MAX_SEQUENCE_LENGTH:
        split_text = t.split('|')

        texts1.append(split_text[0])

        texts2.append(split_text[1])

f.close()
print (get_no_match_texts(argv, texts1))
