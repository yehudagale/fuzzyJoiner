from sys import argv
def proccess(line):
    return line[line.find(">") + 1:line.find("</")]
input_file = open(argv[1])
output_file = open(argv[2], "w")
printMode = False
line_counter = 0
temp_name_store = ""
for line in input_file.readlines():
    if printMode:
        line_counter += 1
    if line.strip().startswith("<lei:OtherEntityNames>"):
        printMode = True
        line_counter = 0
    elif line.strip().startswith("</lei:OtherEntityNames>"):
        printMode = False
        if line_counter > 2:
            output_file.write((temp_name_store + "\n"))
            #print (temp_name_store)
        temp_name_store = ""
    if printMode and line_counter == 1:
        temp_name_store = proccess(line)
    if printMode and line_counter > 1:
        temp_name_store += " | " + proccess(line)
output_file.close()
input_file.close()