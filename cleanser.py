from sys import argv
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-f', dest="input_file or directory", help="file to cleanse file")
    parser.add_argument('-o', dest="output_file", help="cleansed data file name")
    parser.add_argument('-t', dest="entity_type", help="names or people")
 	parser.add_argument('-u', dest="function", help="use test parsing function")
 	parser.add_argument('-n', dest="number", help="process only this many from the file (useful for debugging)")

    args = parser.parse_args()
    if args.entity_type == 'names':
	    cleaner = NameDataCleanser(number)
	else:
		cleaner = GenericDataCleanser(entity_type, function, number)

    cleaner.parse_file(args.filename, args.output, args.rejects)

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

cleaner = GenericDataCleanser()
cleaner.clean_file(argv)