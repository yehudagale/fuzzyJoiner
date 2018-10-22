from sys import argv
onlyfiles = [f for f in listdir(args[1]) if isfile(join(args[1], f))]
for file_path in onlyfiles:
	input_file = open(args[1] + "/" + file_path, encoding='utf-8')
	output_file = open(argv[2] + "/" + file_path, "w", encoding='utf-8')
	write_line = False
	for line in input_file:
		if write_line:
			items = line.split("	")
			output_file.write(items[2] + " | " + items[20] + "\n")
		else:
			write_line = True
	f.close()
	output_file.close()