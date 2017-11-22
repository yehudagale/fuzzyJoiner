from sys import argv
f = open(argv[1])
output_file = open(argv[2], "w")
write_line = False
for line in f:
	if write_line:
		items = line.split("	")
		output_file.write(items[2] + " | " + items[20] + "\n")
	else:
		write_line = True
f.close()
output_file.close()