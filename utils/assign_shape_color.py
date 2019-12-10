#!/usr/bin/env python3
## DESCRIPTION ##
#   the script is used to automatically add color and shape to the barcode info file
#   shape index can be referred at https://www.datanovia.com/en/blog/ggplot-point-shapes-best-tips/

import sys

if len(sys.argv) != 2:
	sys.exit('python3 %s <barcode.txt>' % (sys.argv[0]))

shape = range(0,19)
color = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']

combination = []
for i in shape:
	for j in color:
		combination.append('%s\t%i' % (j, i))

idx = 0
with open(sys.argv[1]) as f:
	for line in f:
		line = line.rstrip()
		tmp = line.split('\t')
		print('%s\t%s' % (line, combination[idx]))
		idx += 1
		