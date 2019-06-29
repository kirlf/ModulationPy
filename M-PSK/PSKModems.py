import numpy as np
import sys
from time import time
import random
import matplotlib.pyplot as plt

class Modem:
    def __init__(self, M, phi, SymMap='Gray', InType='Binary'):
        self.M = M
        self.phi = phi
        if InType != 'Decimal' and InType != 'Binary':     
            print("Wrong input data type (should be 'Decimal' or 'Binary').\n Now InType = " \
                  + str(InType))
            sys.exit(0)    
        if SymMap != 'Gray' and SymMap != 'Binary':     
            print("Wrong mapping type (should be 'Gray' or 'Binary').\n Now SymMap = " \
                  + str(SymMap))
            sys.exit(0)
        self.SymMap = SymMap
        self.InType = InType
        if InType == 'Binary':
            self.BinIn = True
        else:
            self.BinIn = False
    
    def gray_encoder(self, x):
        for idx in range(len(x)):
            if idx == 0:
                y = x[idx]
            else:
                y = y + str ( int( x[idx] ) ^ int( x[idx-1] ) )
        return y
    
    def __gray_encoding(self, s):
        s2 = []
        for i in s:
            symbol = bin(i)[2:]
            if len(symbol) < np.log2(self.M):
                symbol = int( (np.log2(self.M) - len(symbol)) )*'0'+symbol
            s2.append( int(self.gray_encoder(symbol), 2 ) )
        return s2
    
    def __dict_make(self, s, m):
        dict_out = {}
        for x, y in zip(s, m):
            dict_out[x] = y
        return dict_out
    
    def de2bin(self, s):
        b = []
        for i in s:
            a = bin(i)[2:]
            if len(a) < np.log2(self.M):
                a = int((np.log2(self.M) - len(a)))*'0'+a
            if np.log2(self.M)%2 == 0:
                a = a[::-1]
            b.append(a)
        return b
    
    def create_constellation(self, s, m, mode='Modulator'):
        dict_out = {}
        if mode == 'Modulator':
            if self.InType == 'Decimal' and self.SymMap == 'Binary':
                dict_out = self.__dict_make(s, m)
            elif self.InType == 'Decimal' and self.SymMap == 'Gray':
                s2 = self.__gray_encoding(s)
                dict_out = self.__dict_make(s2, m)
            elif self.InType == 'Binary' and self.SymMap == 'Binary':
                b = self.de2bin(s)
                dict_out = self.__dict_make(b, m)
            elif self.InType == 'Binary' and self.SymMap == 'Gray':
                s2 = self.__gray_encoding(s)
                b = self.de2bin(s2)
                dict_out = self.__dict_make(b, m)
        elif mode == 'Demodulator':
            if self.SymMap == 'Binary':
                dict_out = self.__dict_make(s, m)
            elif self.SymMap == 'Gray':
                s2 = self.__gray_encoding(s)
                dict_out = self.__dict_make(s2, m)
        else:
            print("Wrong mode (should be 'Modulator' or 'Demodulator').\n Now mode = " \
                  + str(mode))
            sys.exit(0)
        return dict_out
    
    def ApproxLLR(self, x, zeros, ones, NoiseVar=1):
        LLR = []
        for d in range(len(zeros)): #or for d in range(len(ones)):
            num = []
            for z in zeros[d]:
                num.append( list( ( ( np.real(x) - np.real(z) )**2 ) + ( (np.imag(x) - np.imag(z))**2 ) ) )
            denum = []
            for o in ones[d]:
                denum.append( list( ( ( np.real(x) - np.real(o) )**2 ) + ( (np.imag(x) - np.imag(o))**2 ) ) )
            num_post = np.amin(num, axis=0, keepdims=True)
            denum_post = np.amin(denum, axis=0, keepdims=True)
            llr = np.transpose(num_post[0]) - np.transpose(denum_post[0])
            LLR.append(-llr/NoiseVar)
        result = np.zeros((len(x)*len(zeros))) 
        for i, n in enumerate(LLR):
            result[i::len(zeros)] = n
        return result
    
    def ExactLLR(self, x, zeros, ones, NoiseVar=1):
        LLR = []
        for d in range(len(zeros)): #or for d in range(len(ones)):
            num = []
            for z in zeros[d]:
                num.append( list ( np.exp ( -1* ( ( ( np.real(x) - np.real(z) )**2 )\
                                                 + ( (np.imag(x) - np.imag(z))**2 ) ) / NoiseVar ) ) )
            denum = []
            for o in ones[d]:
                denum.append( list ( np.exp ( -1*  ( ( ( np.real(x) - np.real(o) )**2 )\
                                                + ( (np.imag(x) - np.imag(o) )**2 ) ) / NoiseVar ) ) )
            
            num_post = np.sum(num, axis=0, keepdims=True)
            denum_post = np.sum(denum, axis=0, keepdims=True)
            llr = np.log(num_post / denum_post)
            LLR.append(llr)
        result = np.zeros((len(x)*len(zeros))) 
        for i, n in enumerate(LLR):
            result[i::len(zeros)] = n
        return result
    
    def plot_const(self):
        const = self.create_constellation(self.s, self.m)
        fig = plt.figure(figsize=(6, 4), dpi=150)
        for i in list(const):
            x = np.real(const[i])
            y = np.imag(const[i])
            plt.plot(x, y, 'o', color='green')
            if x < 0:
                h = 'right'
                xadd = -.03
            else:
                h = 'left'
                xadd = .03
            if y < 0:
                v = 'top'
                yadd = -.03
            else:
                v = 'bottom'
                yadd = .03
            if (abs(x) < 1e-9 and abs(y) > 1e-9):
                h = 'center'
            elif abs(x) > 1e-9 and abs(y) < 1e-9:
                v = 'center'     
            plt.annotate(i,(x+xadd,y+yadd), ha=h, va=v)
        if self.M == 2 and self.Type == 'PSK':
            M = 'B'
        elif self.M == 4 and self.Type == 'PSK':
            M = 'Q'
        else:
            M = self.M
        plt.grid()
        plt.axvline(linewidth=1.0, color='black')
        plt.axhline(linewidth=1.0, color='black')
        plt.axis([-1.5,1.5,-1.5,1.5])
        plt.title(str(M)+'-'+str(self.Type)+', phase rotation: '+str(round(self.phi, 5))+\
                  ', Mapping: '+str(self.SymMap)+', Input: '+str(self.InType))
        plt.show()

class PSKModulator(Modem):  
    def __init__(self, M, phi, SymMap='Gray', InType='Binary'):
        super().__init__(M, phi, SymMap, InType)
        self.s = [i for i in range(self.M)]
        self.m = list(np.exp(1j*self.phi + 1j*2*np.pi*np.array(self.s)/self.M))
        self.Type = 'PSK'
        self.code_book = self.create_constellation(self.s, self.m)
    
    def __fast_qpsk_mod(self, s):
        m = (s[::2]*(-2)+1)*np.cos(np.pi/4)+1j*(s[1::2]*(-2)+1)*np.sin(np.pi/4)
        return m       
                                    
    def modulate(self, x):
        modulated = []
        if self.M == 4 and self.phi == np.pi / 4 and self.SymMap=='Gray' and self.InType=='Binary':
            modulated = self.__fast_qpsk_mod(x)
        else:
            if self.BinIn == True: 
                m = []
                n = int(np.log2(self.M))
                length = len(x)
                for c in range(int(length/n)):
                    s = ''
                    y = x[(c + (n - 1)*c):(((n - 1)*c) + (n - 1) + (1+c))]
                    for d in y:
                        s = s+str(int(d))
                    modulated.append(self.code_book[s])
            else:
                for a in x:
                    modulated.append(self.code_book[a])
        return np.array(modulated)

class PSKDemodulator(Modem):
    def __init__(self, M, phi, SymMap='Gray', InType='Binary', DecisionMethod='Approximate LLR'):
        super().__init__(M, phi, SymMap, InType)
        
        zeros = []
        ones = []
        for c in range(int(np.log2(self.M))):
            zeros.append([])
            ones.append([])
        s = [i for i in range(self.M)]
        m = list(np.exp(1j*self.phi + 1j*2*np.pi*np.array(s)/self.M))
        codebook = self.create_constellation(s, m, mode='Demodulator')
        b = self.de2bin(s)
        for idx, n in enumerate(b):
            for ind, m in enumerate(n):
                if m == '0':
                    zeros[ind].append(codebook[idx])
                else:
                    ones[ind].append(codebook[idx])
        self.zeros = zeros
        self.ones = ones
        self.DecisionMethod = DecisionMethod
        
    def __fast_qpsk_demod(self, x):
        LLR = []
        for inx in x:
            re =  (-( np.real(inx) - np.cos(np.pi/4))**2 ) - ( -(np.real(inx) + np.cos(np.pi/4))**2 )
            im =  (-( np.imag(inx) - np.sin(np.pi/4))**2 ) - ( -(np.imag(inx) + np.sin(np.pi/4))**2 )
            LLR.append(float(re))
            LLR.append(float(im))
        return np.array(LLR)        
    
    def demodulate(self, x):
        if self.M == 4 and self.phi == np.pi / 4 and self.SymMap=='Gray' and self.InType=='Binary':
            if self.DecisionMethod == 'Approximate LLR' or self.DecisionMethod == 'Exact LLR':
                result = self.__fast_qpsk_demod(x)
            elif self.DecisionMethod == 'Hard':
                result = (np.sign(-self.__fast_qpsk_demod(x)) + 1) / 2
        else:
            if self.DecisionMethod == 'Exact LLR':
                result = self.ExactLLR(x, self.zeros, self.ones)
            elif self.DecisionMethod == 'Approximate LLR':
                result = self.ApproxLLR(x, self.zeros, self.ones)
            elif self.DecisionMethod == 'Hard':
                result = (np.sign(-self.ApproxLLR(x, self.zeros, self.ones)) + 1) / 2
            else:
                print("Wrong Decision Method (should be Approximate LLR, Exact LLR or Hard). Now Decision Method = "\
                      + str(self.DecisionMethod))
                sys.exit(0)                            
        return result  
