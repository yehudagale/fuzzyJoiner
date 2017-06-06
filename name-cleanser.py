from sys import argv
from string import maketrans   # Required to call maketrans function.
reject_set = frozenset(["Father of the", "("])
def good_data(data):
	#used https://www.tutorialspoint.com/python/string_translate.htm
	#change if we want to preserve any of these things
	intab = "\n"
	outtab = " "
	trantab = maketrans(intab, outtab)
	data.translate(trantab)
	try:
		data.encode('utf-8')
	except UnicodeDecodeError:
		return False
	data_words = data.split()
	for item in reject_set:
		if item in data:
			return False
	return True
#cleanses the data and returns only good data
def cleanseData(dataToCleanse):
	ret = []
	name_array = dataToCleanse.split("|")
	for part in name_array:
		if len(ret) >= 2:
			break
		if good_data(part):
			if part[-1:] == "\n":
				part = part[:-1]
			ret.append(part)
	if len(ret) == 2:
		return "|".join(ret)
	return ""
#parses input file and writes an output
def parse_file(input_file, output_file):
	lines = input_file.readlines()
	for line in lines:
		newline = cleanseData(line)
		if newline != "":
			output_file.write(newline)
			output_file.write('\n')

if len(argv) == 3:
	input_file = open(argv[1])
	output_file = open(argv[2], "w")
	parse_file(input_file, output_file)
	input_file.close()
	output_file.close()

