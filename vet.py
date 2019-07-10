# coding: utf-8
#import numpy as np
import itertools
import math
sqrt = math.sqrt
#corr = np.corrcoef
INF = 1e-20

def _sum(v):
	s = 0
	for i in v: s += i
	return s
	
p1 = (1, 3)
p2 = (2, 1)
def dist(p1, p2):
	return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) 
	 
def	distV(v1, v2):
	s = 0
	for i in range(len(v1)):
		s += (v2[i] - v1[i]) ** 2
	return sqrt(s)

def mid(v): return _sum(v) / len(v)

def _min(v):
	min = v[0]
	for i in v:
		if min > i:
			min = i
	return min
	
def _max(v):
	max = v[0]
	for i in v:
		if max < i:
			max = i
	return max
	
def desv(v):
	s = 0
	for i in v:
		s += (i - mid(v)) ** 2
	return sqrt(s / len(v))

def corr(v1, v2):
	s1 = s2 = s3 = 0
	for i in range(len(v1)):
		s1 += (v1[i] - mid(v1)) * (v2[i] - mid(v2))
		s2 += (v1[i] - mid(v1)) ** 2
		s3 += (v2[i] - mid(v2)) ** 2
	return s1 / sqrt(s2 * s3)

def dot(v1, v2):
	s = 0
	for i in range(len(v1)): s += v1[i] * v2[i]
	return s
	
def dot2dif(v1, v2):
	s = 0
	for i in range(len(v1)):
		s += (v1[i] - v2[i]) ** 2
	return s

def cos_v(v1, v2): return dot(v1, v2) / (INF + (sqrt(dot(v1, v1))) * (sqrt(dot(v2, v2))))

def dist_euc_v(v1, v2): return sqrt(dot2dif(v1, v2))

def corr_v(v1, v2):
	l = len(v1); s1 = sum(v1); s2 = sum(v2)
	a = (INF + (sqrt(l * dot(v1, v1) - s1 ** 2)) * (sqrt(l * dot(v1, v1) - s1 ** 2)))
	return (l * dot(v1, v2) - s1 * s2) / a

