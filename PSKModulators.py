#!/usr/bin/python

import numpy as np
import sys

def fast_qpsk_mod(s):
	m = (s[::2]*(-2)+1)*np.cos(np.pi/4)+1j*(s[1::2]*(-2)+1)*np.sin(np.pi/4)
	return m


def GRAYencoder(x):
	for idx in range(len(x)):
		if idx == 0:
			y = x[idx]
		else:
			y = y + str ( int( x[idx] ) ^ int( x[idx-1] ) )
	return y




def QPSKmodulator(phase_shift = np.pi/4, input_type = 'Gray'):
	s = [0+i for i in range(4)]
	dict_out = {}
	if input_type == 'Binary':
		m = list(np.exp(1j*phase_shift + 1j*2*np.pi*np.array(s)/4))
		for x, y in zip(s, m):
			dict_out[x] = y
	elif input_type == 'Gray':
		s2 = []
		for i in s:
			symbol = bin(i)[2:]
			if len(symbol) < np.log2(4):
				symbol = '0'+symbol
			s2.append( int( symbol[0] + str( int( symbol[0] ) ^ int( symbol[1] ) ), 2 ) )
		m = list(np.exp(1j*phase_shift + 1j*2*np.pi*np.array(s2)/4))
		for x, y in zip(s, m):
			dict_out[x] = y
	else:
		print "Wrong iput data type (should be 'Gray' or 'Binary'). Now input_type = "+input_type+""
		sys.exit(0)
	return dict_out



def PSKmodulator(modulation_order, phase_shift = np.pi/4, input_type = 'Gray'):
	s = [0+i for i in range(modulation_order)]
	m = list(np.exp(1j*phase_shift + 1j*2*np.pi*np.array(s)/modulation_order))
	dict_out = {}
	if input_type == 'Binary':
		for x, y in zip(s, m):
			dict_out[x] = y
	elif input_type == 'Gray':
		s2 = []
		for i in s:
			symbol = bin(i)[2:]
			if len(symbol) < np.log2(modulation_order):
				symbol = int( (np.log2(modulation_order) - len(symbol)) )*'0'+symbol
			s2.append( int(GRAYencoder(symbol), 2 ) )
		for x, y in zip(s2, m):
			dict_out[x] = y
	else:
		print "Wrong iput data type (should be 'Gray' or 'Binary'). Now input_type = "+str(input_type)+""
		sys.exit(0)
	return dict_out



def modulate(x, modulation_order, code_book, binary_input = True):
	modulated = []
	if binary_input == True: 
		m = []
		s = ''
		n = int(np.log2(modulation_order))
		length = len(x)
		for c in range(int(length/n)):
			y = x[(c + (n - 1)*c):(((n - 1)*c) + (n - 1) + (1+c))]
			for d in y:
				s = s+str(int(d))
			m.append(int(s[::-1],2))
			s = ''
		for a in m:
			modulated.append(code_book[a])
	elif binary_input == False:
		for a in x:
			modulated.append(code_book[x])
	else:
		print "Wrong data type for binary_input (should be True or False). Now input_type = "+str(input_type)+""		
	return np.array(modulated)
