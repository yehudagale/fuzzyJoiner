from sys import argv
from random import shuffle
input_file = open(argv[1], 'r')
output_file = open(argv[2], 'w')
lines = input_file.readlines()
shuffle(lines)
lines = lines[-int(argv[3]):]
for line in lines:
	output_file.write(line)
input_file.close()
output_file.close()
