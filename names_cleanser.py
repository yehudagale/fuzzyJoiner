import re
import argparse
import Levenshtein
import nltk
from nltk import bigrams

class name_data_cleanser(object):

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

		arr = line.split("|")


		# remove silly names
		cleansed_arr = []
		for name in arr:
			# remove all abbreviated names that end with .
			if name.endswith('.'):
				continue
			# remove all single names
			if len(name.split(' ')) == 1:
				continue
			cleansed_arr.append(name)

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

		base = cleansed_arr[0].lower().replace(' ', '')
		# print(cleansed_arr[0])
		for j in range(1, len(cleansed_arr)):
			cmp = cleansed_arr[j].lower().replace(' ', '')
			ratio = Levenshtein.ratio(base, cmp)
			if (ratio == 0):
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

		if len(ret) == 1:
			line = cleansed_arr[0].replace('\n','')
		else:
			# put the first guy in first... or else we lose track of the 'anchor'
			ret.remove(cleansed_arr[0])
			line = cleansed_arr[0] + '|'
			line += '|'.join(ret)
			line = line.replace('\n', '')

		return line


	def parse_file(self, input_file, output_file, output_rejects_file):
		out = open(output_file, 'w')
		reject = open(output_rejects_file, 'w')
		with open(input_file) as f:
			for line in f:
				newline = self.cleanse_data(line)
				if newline and newline != "":
					out.write(line)
					out.write(newline)
					out.write('\n')
				else:
					reject.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-f', dest="filename", help="file to cleanse file")
    parser.add_argument('-o', dest="output", help="cleansed data file name")
    parser.add_argument('-r', dest="rejects", help="rejected data file")
    args = parser.parse_args()
    cleaner = name_data_cleanser()
    cleaner.parse_file(args.filename, args.output, args.rejects)
    

