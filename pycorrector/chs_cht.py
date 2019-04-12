# -*- coding: utf8 -*-
import sys
from opencc import OpenCC

cc = OpenCC('s2t')

def simplify(text):
 return cc.convert(text)#.encode('utf8')

def traditionalize(text):
 return cc.convert(text)#.encode('utf8')

if len(sys.argv) < 2:
	exit()

for l in open(sys.argv[1], 'r'):
	print(traditionalize(l), end='')

