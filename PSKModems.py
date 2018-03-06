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


def PSKmodulator(modulation_order, phase_shift = np.pi/4, symbol_mapping = 'Gray', input_type = 'Binary'):
	s = [0+i for i in range(modulation_order)]
	m = list(np.exp(1j*phase_shift + 1j*2*np.pi*np.array(s)/modulation_order))
	dict_out = {}
	if input_type == 'Decimal':
		if symbol_mapping == 'Binary':
			for x, y in zip(s, m):
				dict_out[x] = y
		elif symbol_mapping == 'Gray':
			s2 = []
			for i in s:
				symbol = bin(i)[2:]
				if len(symbol) < np.log2(modulation_order):
					symbol = int( (np.log2(modulation_order) - len(symbol)) )*'0'+symbol
				s2.append( int(GRAYencoder(symbol), 2 ) )
			for x, y in zip(s2, m):
				dict_out[x] = y
		else:
			print("Wrong input data type (should be 'Gray' or 'Binary'). Now input_type = " + str(input_type))
			sys.exit(0)
	else:
		if symbol_mapping == 'Binary':
			b = []
			for i in s:
				a = bin(i)[2:]
				if len(a) < np.log2(modulation_order):
					a = int((np.log2(modulation_order) - len(a)))*'0'+a
				if np.log2(modulation_order)%2 == 0:
					a = a[::-1]
				b.append(a)
			for x, y in zip(b, m):
				dict_out[x] = y
		elif symbol_mapping == 'Gray':
			s2 = []
			for i in s:
				symbol = bin(i)[2:]
				if len(symbol) < np.log2(modulation_order):
					symbol = int( (np.log2(modulation_order) - len(symbol)) )*'0'+symbol
				s2.append( int(GRAYencoder(symbol), 2 ) )
			b = []
			for i in s2:
				a = bin(i)[2:]
				if len(a) < np.log2(modulation_order):
					a = int((np.log2(modulation_order) - len(a)))*'0'+a
				if np.log2(modulation_order)%2 == 0:
					a = a[::-1]
				b.append(a)
			for x, y in zip(b, m):
				dict_out[x] = y
		else:
			print("Wrong input data type (should be 'Gray' or 'Binary'). Now input_type = " + str(input_type))
			sys.exit(0)		
	return dict_out



def modulate(x, modulation_order, code_book, binary_input = True):
	modulated = []
	if binary_input == True: 
		m = []
		n = int(np.log2(modulation_order))
		length = len(x)
		for c in range(int(length/n)):
			s = ''
			y = x[(c + (n - 1)*c):(((n - 1)*c) + (n - 1) + (1+c))]
			for d in y:
				s = s+str(int(d))
			modulated.append(code_book[s])
	elif binary_input == False:
		for a in x:
			modulated.append(code_book[x])
	else:
		print("Wrong data type for binary_input (should be True or False). Now input_type = " + str(input_type))		
	return np.array(modulated)


def PSKDemodulator(modulation_order, phase_shift = np.pi/4, symbol_mapping = 'Gray'):
	zeros = []
	ones = []
	for c in range(int(np.log2(modulation_order))):
		zeros.append([])
		ones.append([])
	codebook = PSKmodulator(modulation_order, phase_shift, symbol_mapping, 'Decimal')
	s = [0+i for i in range(modulation_order)]
	b = []
	for i in s:
		a = bin(i)[2:]
		if np.log2(modulation_order)%2 == 0:
			a = a[::-1]
		if len(a) < np.log2(modulation_order):
			a = int((np.log2(modulation_order) - len(a)))*'0'+a
		b.append(a)
	for idx, n in enumerate(b):
		for ind, m in enumerate(n):
			if m == '0':
				zeros[ind].append(codebook[idx])
			else:
				ones[ind].append(codebook[idx])
	return zeros, ones
		
	
def ExactLLR(x, zeros, ones):
	LLR = []
	for c in range(len(x)):
		for d in range(len(zeros)):
			num = 0
			for z in zeros[d]:
				num =  num + ( np.exp ( -  ( ( ( np.real(x[c]) - np.real(z) )**2 ) + ( (np.imag(x[ c ]) - np.imag(z))**2 ) ) ) )
			denum = 0
			for o in ones[d]:
				denum =  denum + ( np.exp ( -  ( ( ( np.real(x[c]) - np.real(o) )**2 ) + ( (np.imag(x[c]) - np.imag(o))**2 ) ) ) )
			llr = np.log( num / denum )
			LLR.append(llr) 
	return LLR



def ApproxLLR(x, zeros, ones):
	LLR = []
	for d in range(len(zeros)):
		num = []
		for z in zeros[d]:
			num.append( list( ( ( np.real(x) - np.real(z) )**2 ) + ( (np.imag(x) - np.imag(z))**2 ) ) )
		denum = []
		for o in ones[d]:
			denum.append( list( ( ( np.real(x) - np.real(o) )**2 ) + ( (np.imag(x) - np.imag(o))**2 ) ) )
		num_post = num[0]
		for n in num:
			num_post = np.minimum(num_post, n)
		denum_post = denum[0]
		for m in denum:
			denum_post = np.minimum(denum_post, m)
		llr = np.transpose(np.array(num_post)) - np.transpose(np.array(denum_post))
		LLR.append(-llr)
	result = np.zeros((len(x)*len(zeros))) 
	for i, n in enumerate(LLR):
		result[0+i::len(zeros)] = n
	return result

