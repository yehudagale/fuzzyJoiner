import re
import argparse
import Levenshtein
from nltk import bigrams
from os import listdir
from os.path import isfile, join
from sklearn.metrics import jaccard_similarity_score


class GenericDataCleanser(object):

    name_reject_set = frozenset(['father of', '(', 'author of', 'pope', 'emperor'])
    company_reject_set = frozenset(["("])

    def __init__(self, entity_type, function = None, number=None, pairs=2):
        if number:
            self.get=True
            self.number_of_names = number
        else:
            self.get = False

        function_dictionary = {"nametest":self.test_x_names, "companytest":self.test_x_companies, "name":self.good_name_data, "company":self.good_company_data, "names":self.good_name_data}
        if entity_type == "name" or entity_type == "names" or entity_type == "company":
            self.parsing_function = function_dictionary[entity_type]
        elif function and function == "test":
            self.parsing_function = function_dictionary[entity_type + "test"]
        self.pairs = 2

    def test_x_names(self, data):
        self.number_of_names -= 1
        if self.number_of_names < 0:
            return False
        else:
            return self.good_name_data(data)

    def test_x_companies(self, data):
        self.number_of_names -= 1
        if self.number_of_names < 0:
            return False
        else:
            return self.good_company_data(data)

    def is_english(self, data):
        try:
            data.encode('ascii')
        except UnicodeEncodeError:
            return False
        return True

    def fix_bad_chars(self, data):
        #used https://www.tutorialspoint.com/python/string_translate.htm
        #change if we want to preserve any of these things
        intab = "\n"
        outtab = " "
        trantab = str.maketrans(intab, outtab)
        return data.translate(trantab)

    #test whether or not a company name should be used
    def remove_bad(self, data):
        data = data.replace('"', "(")
        data = data.replace("\n", "")
        data = data.replace("  ", "")
        return data.lstrip()

    def good_company_data(self, data):
        if data.startswith("<http://dbpedia.org/resource"):
            return False
        if data.startswith("The Master Trust Bank of Japan"):
            return False
        if not self.is_english(data):
            return False
        data = self.fix_bad_chars(data)
        data = data.lower()
        for item in self.company_reject_set:
            if item in data:
                return False
        return True

    def good_name_data(self, data):
        if not self.is_english(data):
            return False
        data = self.fix_bad_chars(data)
        data = data.lower()
        for item in self.name_reject_set:
            if item in data:
                return False
        return True

    #cleanses the data and returns only good data
    #if we cannot get 2 peices of good data returns the empty string
    def cleanse_data(self, dataToCleanse):
        ret = []
        name_array = dataToCleanse.split("|")
        for part in name_array:
            if not part:
                continue
            if len(ret) >= self.pairs:
                break
            if self.parsing_function(part):
                # if part[-1:] == "\n":
                # 	part = part[:-1]
                if part not in ret:
                    part = self.remove_bad(part)
                    if part:
                        part = part.replace(',', ' , ').replace('-', ' - ').replace('.', ' . ').replace('  ', ' ')
                        ret.append(part)
        if len(ret) != self.pairs:
            return None
        return ret

    #parses input file and writes an output
    def parse_file(self,input_file, output_file, output_rejects_file):
        lines = input_file.readlines()
        entity_id = 0
        print('in parse file')
        print(self.pairs)
        for line in lines:
            ret = self.cleanse_data(line)
            if ret and len(ret) >= self.pairs:
                # all these entity names are in the exact same set
                # we can create the number of pairs that we were asked to create
                # for now, create it with the 'anchor' (first element of the list)
                # and all other names
                anchor = ret[0]
                entity_id += 1
                newline =''
                for i in range(1, len(ret)):
                    newline = anchor + '|' + ret[i] + "|" + str(entity_id) + '\n'
                    if self.get:
                        self.number_of_names -= 1
                        if self.number_of_names >= 0:
                            output_file.write(newline)
                        else:
                            self.get = False;
                            break
                    else:
                        output_file.write(newline)
                    if i > self.pairs:
                        break
            else:
                output_rejects_file.write(line)
        if self.get:
            print("ran out of names, proceeding with as many as were available")

    def clean_file(self, filename, output):
        onlyfiles = [f for f in listdir(filename) if isfile(join(filename, f))]
        output_file = open(output, "w", encoding='utf-8')
        output_rejects_file = open('./rejects.txt', 'w', encoding='utf-8')
        output_file = open(output, "w", encoding='utf-8')
        for file_path in onlyfiles:
            input_file = open(filename + "/" + file_path, encoding='utf-8')
            self.parse_file(input_file, output_file, output_rejects_file)
        output_rejects_file.close()
        input_file.close()
        output_file.close()


class NameDataCleanser(GenericDataCleanser):

    MAX_PAIRS = 6

    def __init__(self, number=0, pairs = 2, limit_pairs = True):
        if number > 0:
            self.get=True
            self.number_of_names = number
        else:
            self.get = False
        self.pairs = pairs
        self.limit_pairs = limit_pairs

    def create_new_name(self, name, current_name_set):
        names = name.split()
        ret = set([])

        if len(names) <= 2:
            ret.add(names[1] + ' , ' + names[0])		# last_name, first_name
            ret.add(names[0][0] + ' . ' + names[1])		# first_initial. lastname
            ret.add(names[1] + ' , ' + names[0][0])		# last_name, first_initial.
        elif len(names) > 2:
            ret.add(names[0] + ' ' + names[1][0] + ' . ' + names[-1]) # first_name middle initial last_name
            ret.add(names[-1] + ' , ' + names[0] + ' ' + names[1])		# last_name, first name, middle name
            ret.add(names[-1] + ' , ' + names[0] + ' ' + names[1][0] + ' . ') # last_name, first name, middle initial
            ret.add(names[-1] + " " + names[0])		# drop all middle names
            ret.add(names[0][0] + ' . ' + names[-1])	# first initial. lastname
            ret.add(names[-1] + ' , ' + names[0][0])		# last_name, first_initial.

        diff = ret.difference(set(current_name_set))
        return diff

    def cleanse_data(self, line):
        # skip lines with pope/emperor/queen etc they tend to be useless
        titles = ['king','queen','emperor']
        qualifiers = ['of', 'I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII']
        if 'pope' in line.lower():
            return
        for p in titles:
            if p in line.lower():
                for q in qualifiers:
                    if q in line:
                        return


        # remove any names with "..."  with nothing. also remove all (.*)
        line = re.sub('["][^"]*["]', '', line)
        line = re.sub('[(][^)]*[)]', '', line)
        # remove any non-Latin characters from the line
        line = re.sub('[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]','',line)
        arr = line.split("|")

        dups = []
        for name in arr:
            dups.append(name.replace(',', ' , ').replace('-', ' - ').replace('.', ' . ').replace('  ', ' ').strip())

        if (len(set(dups))) == 1:
            return

        # remove silly names
        cleansed_arr = []
        for name in arr:
            if name.strip() == '' or name.strip() == '.':
                continue
            # remove all abbreviated names that end with .
            if name.endswith('.'):
                continue
            # remove all single names
            if len(name.split(' ')) == 1:
                continue

            name = name.replace(',', ' , ').replace('-', ' - ').replace('.', ' . ').replace('  ', ' ').strip()

            cleansed_arr.append(name.replace('\n', '').replace('"',''))

        if len(cleansed_arr) == 0 or len(cleansed_arr[0].strip().split(' ')) == 1:
            return

        # check if the name is simply some name with all other words is just a title
        base_parts_set = set(cleansed_arr[0].split())
        if len(base_parts_set.intersection(set(qualifiers))) == 1:
            return


        # compare each name with every other name in the array to make sure
        # we have at least one name part in common with the first name, which we assume is
        # the 'canonical' name.
        ret = set(cleansed_arr)

        base = cleansed_arr[0].lower()
        # print(cleansed_arr[0])
        for j in range(1, len(cleansed_arr)):
            cmp = cleansed_arr[j].lower()
            # same name just changed case
            if base == cmp and cleansed_arr[j] in ret:
                ret.remove(cleansed_arr[j])
                continue
            ratio = Levenshtein.ratio(base, cmp)
            if (ratio == 0 and cleansed_arr[j] in ret):
                ret.remove(cleansed_arr[j])
                # print('removing due to ratio' + cleansed_arr[j] + "|" + cleansed_arr[0])
                continue
            # string similarity metrics will tell you Richard Phillips is similar to Jack Dowling because they look at it character by character
            # need something at a word level.  Look at all the name parts in the thing we are comparing to base, and make sure at least
            # one name overlaps
            match = False
            for part in cleansed_arr[j].lower().split(' '):
                part = part.replace('\n','')
                if part in base:
                    match = True

            if not match and cleansed_arr[j] in ret:
                ret.remove(cleansed_arr[j])
                # print('removing due to no overlap in names' + cleansed_arr[j] + "|" + cleansed_arr[0])
                continue


            # often we have situations where the first name matches but everything else does not.  This happens
            # when a woman changes names for instance.  A complication here is that Japanese names reverse names
            # so we need to not remove that case
            match = False
            s = set(cleansed_arr[0].lower().split()).intersection(set(cleansed_arr[j].lower().split()))
            if len(s) > 1:
                match = True

            rest_of_base = ''.join(cleansed_arr[0].lower().split()[1:])
            compare_to = ''.join(cleansed_arr[j].lower().split()[1:])
            base_bigrams = set(bigrams(rest_of_base))
            cmp_bigrams = set(bigrams(compare_to))

            if len(base_bigrams.intersection(cmp_bigrams)) == 0 and not match:
                # print(base_bigrams)
                # print(cmp_bigrams)
                if cleansed_arr[j] in ret:
                    # print('removing due to looks like a female name changed after marriage' + cleansed_arr[j] + base)
                    ret.remove(cleansed_arr[j])
                continue

        anchor = cleansed_arr[0]

        ret_val = [cleansed_arr[0]]

        # put anchor in first, then add all other elements
        for n in ret:
            if n != cleansed_arr[0]:
                ret_val.append(n)

        if len(ret_val) < self.pairs:
            additional_elements = self.create_new_name(anchor, ret_val)
            ret_val.extend(additional_elements)
            if len(ret_val) < self.pairs:
                return
        elif len(ret_val) > self.MAX_PAIRS:
            anchor_parts = set(anchor.split())

            def jaccard_sim_function(x):
                x_parts = set(x.split())
                return len(anchor_parts.intersection(x_parts)) / len(anchor_parts.union(x_parts))

            sorted(ret_val, key=jaccard_sim_function)
            # print(ret_val)
            ret_val = ret_val[0:self.MAX_PAIRS]

        assert len(set(ret_val)) >= self.pairs
        return ret_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', dest="input_file", help="file to cleanse file")
    parser.add_argument('-o', dest="output_file", help="cleansed data file name")
    parser.add_argument('-t', dest="entity_type", help="names or companies")
    parser.add_argument('-p', dest="num_pairs", help="how many same pairs should be created (default=2)", nargs='?', default=2, type=int)
    parser.add_argument('-u', dest="function", help="use test parsing function", nargs='?', default=None)
    parser.add_argument('-n', dest="number", help="process only this many from the file (useful for debugging)", nargs='?', type=int, default = 0)
    parser.add_argument('-l', dest="no_limit", nargs='?', type=bool, default=False)

    args = parser.parse_args()
    assert args.num_pairs <= 4
    print(args.num_pairs)

    if args.entity_type == 'names':
        cleaner = NameDataCleanser(args.number, args.num_pairs)
    else:
        cleaner = GenericDataCleanser(args.entity_type, args.function, args.number, args.num_pairs)

    # Uncomment next line to test old code
    # cleaner = GenericDataCleanser(args.entity_type, args.function, args.number, args.num_pairs)
    cleaner.clean_file(args.input_file, args.output_file)
    

