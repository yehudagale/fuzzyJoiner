from sys import argv
#used https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
from os import listdir
from os.path import isfile, join
#from str import maketrans   # Required to call maketrans function.
#first use a set of things we don't want in the final output
#make sure encoding is set correctly

class data_cleanser(object):

	"""docstring for data_cleanser"""
	name_reject_set = frozenset(["father of", "(", "author of"])
	company_reject_set = frozenset(["("])
	def __init__(self):
		self.get = False
		self.function_dictionary = {"nametest":self.test_x_names, "companytest":self.test_x_companies, "name":self.good_name_data, "company":self.good_company_data, "names":self.good_name_data}
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
	def cleanseData(self, dataToCleanse, cleansing_function):
		ret = []
		name_array = dataToCleanse.split("|")
		for part in name_array:
			if not part:
				continue
			if len(ret) >= 2:
				break
			if cleansing_function(part):
				# if part[-1:] == "\n":
				# 	part = part[:-1]
				if part not in ret:
					part = self.remove_bad(part)
					if part:
						ret.append(part)
		if len(ret) == 2:
			return "|".join(ret)
		#global test_file
		#test_file.write(dataToCleanse)
		return ""
	#parses input file and writes an output
	def parse_file(self,input_file, output_file, parsing_function, output_rejects_file):
		lines = input_file.readlines()
		for line in lines:
			newline = self.cleanseData(line, parsing_function)
			if newline != "":
				if self.get:
					self.number_of_names -= 1
					if self.number_of_names >= 0:
						output_file.write(newline)
						output_file.write('\n')
					else:
						self.get = False;
						break
				else:
					output_file.write(newline)
					output_file.write('\n')
			else:
				output_rejects_file.write(line)
				output_rejects_file.write('\n')
		if self.get:
			print ("ran out of names, procced as many as were available")


	def clean_file(self, args):
		if len(args) >= 3:
			parsing_function = self.good_name_data
			if len(args) >= 4:
				parsing_function = self.function_dictionary[args[3]]
			if len(args) >= 6:
				self.number_of_names = int(args[5])
				if args[4] == "test":
					parsing_function = self.function_dictionary[(args[3] + args[4])]
				elif args[4] == "get":
					self.get = True
			onlyfiles = [f for f in listdir(args[1]) if isfile(join(args[1], f))]
			output_file = open(args[2], "w", encoding='utf-8')
			output_rejects_file = open('./rejects.txt', 'w', encoding='utf-8')
			output_file = open(args[2], "w", encoding='utf-8')
			for file_path in onlyfiles:
				input_file = open(args[1] + "/" + file_path, encoding='utf-8')
				self.parse_file(input_file, output_file, parsing_function, output_rejects_file)
			output_rejects_file.close()
			input_file.close()
			#test_file.close();
			output_file.close()
		else:
			print ("too few arguments please enter arguments in the following format: input_file output_file [function [get x OR test x]]")
cleaner = data_cleanser()
cleaner.clean_file(argv)