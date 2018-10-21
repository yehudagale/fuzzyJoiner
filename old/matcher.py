#using tutorial https://suhas.org/sqlalchemy-tutorial/
from sys import argv
from matcher_functions import *
#establish connection to database
con, meta = connect(argv[1], argv[2], argv[3])
#load pairs from database
aliases = get_aliases(con, meta)
#create dictionaries assingning serial numbers to names and names from serial numbers
num_to_word, word_to_num = create_double_num_dicts(aliases)
#load the buckets from the database bucket_list is aranges as follows:
#bucket_list[pair_of_buckets][bucket(this must be 0 or 1)][name (this represents a single name)][0 for number and 1 for pre-procced name]
bucket_list, bucket_words = load_good_buckets('wordtable1', 'wordtable2', word_to_num, con, meta)
#print out the number of names that are possible to get just based on bucketing:
impossible = get_impossible(aliases, bucket_list, num_to_word)
print("possible matches: " + str(len(aliases) - len(impossible)))
#next make a list to store the outcomes of all our tests:
matches_list = []
#then run our tests
matches_list.append(run_test(lambda x : x.replace(" ", ""), lambda name1, name2 : name1 in name2 or name2 in name1, num_to_word, bucket_list))
matches_list.append(run_test(lambda x : set(x.split()), lambda name1, name2 : name1.issubset(name2) or name2.issubset(name1), num_to_word, bucket_list))
matches_list.append(run_special_test(bucket_list, num_to_word))
#next create a test dictionary relating each item in the first set to k items in other set
test_dict = make_test_dict(set([]).union(*matches_list), 1000)
#use this dictionary to calculate and print the f-score
print("fscore: " + str(fscore(aliases, test_dict, 1)))
#next export the items we missed
export_missed(aliases, test_dict, con, meta)
#lastly export the items we could not have gotten since they were not in the same bucket:
export_unbucketed(impossible, con, meta)