f_in = open('modal_nano_gpt.py', 'r')
f_out = open('modal_nano_gpt_markdown.py', 'w')

while True:
    line = f_in.readline()
    if line == "" or line == None:
        break
    if line[0] != '#':
        f_out.write('\t' + line)
    else:
        f_out.write(line[1:])
f_in.close()
f_out.close()
