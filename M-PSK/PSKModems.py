import numpy as np
import sys
from time import time
import random
import matplotlib.pyplot as plt

class Modem:
    def __init__(self, M, gray_map=True, bin_input='Binary', decision_method = 'Approximate LLR'):
        
        if np.log2(M) != np.round(np.log2(M)):
            raise ValueError("M should be 2**n, with n=1, 2, 3...")  
        if decision_method != 'Approximate LLR' and decision_method != 'Exact LLR' and decision_method != 'Hard':
            raise ValueError("Wrong Decision Method (should be Approximate LLR, Exact LLR or Hard).\n Now Decision Method = "\
                  + str(decision_method))
        
        self.M = M    
        self.m = [i for i in range(self.M)]    
        self.gray_map = gray_map
        self.bin_input = bin_input
        self.decision_method = decision_method
        
       
    def gray_encoding(self, s):
        s2 = []
        for i in s:
            symbol = bin(i)[2:]
            if len(symbol) < np.log2(self.M):
                symbol = int( (np.log2(self.M) - len(symbol)) )*'0'+symbol
            for idx in range(len(symbol)):
                if idx == 0:
                    y = symbol[idx]
                else:
                    y = y + str(int(symbol[idx])^int(symbol[idx-1]))
            s2.append(int(y, 2))
        return s2

    def dict_make(self, s, m):
        dict_out = {}
        for x, y in zip(s, m):
            dict_out[x] = y
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
    
class PSKModem(Modem):
    def __init__(self, M, phi=0, gray_map=True, bin_input=True, decision_method='Approximate LLR'):
        super().__init__(M, gray_map, bin_input, decision_method)
        self.phi = phi 
        self.s = list(np.exp(1j*self.phi + 1j*2*np.pi*np.array(self.m)/self.M))
        self.code_book = self.__create_constellation(self.m, self.s)
        self.zeros, self.ones = self.__llr_preparation()  
      
    
    def __de2bin(self, s):
        b = []
        for i in s:
            a = bin(i)[2:]
            if len(a) < np.log2(self.M):
                a = int((np.log2(self.M) - len(a)))*'0'+a
            if np.log2(self.M)%2 == 0:
                a = a[::-1]
            b.append(a)
        return b
    
    def __create_constellation(self, s, m, mode='Modulator'):
        dict_out = {}
        if mode == 'Modulator':
            if self.bin_input == False and self.gray_map == False:
                dict_out = self.dict_make(s, m)
            elif self.bin_input == False and self.gray_map == True:
                s2 = self.gray_encoding(s)
                dict_out = self.dict_make(s2, m)
            elif self.bin_input == True and self.gray_map == False:
                b = self.__de2bin(s)
                dict_out = self.dict_make(b, m)
            elif self.bin_input == True and self.gray_map == True:
                s2 = self.gray_encoding(s)
                b = self.__de2bin(s2)
                dict_out = self.dict_make(b, m)
        elif mode == 'Demodulator':
            if self.gray_map == False:
                dict_out = self.dict_make(s, m)
            elif self.gray_map == True:
                s2 = self.gray_encoding(s)
                dict_out = self.dict_make(s2, m)
        return dict_out

    def __llr_preparation(self):
        code_book_demod = self.__create_constellation(self.m, self.s, mode='Demodulator')
        zeros = []  
        ones = []
        for c in range(int(np.log2(self.M))):
            zeros.append([])
            ones.append([])
        b = self.__de2bin(self.m)
        for idx, n in enumerate(b):
            for ind, m in enumerate(n):
                if m == '0':
                    zeros[ind].append(code_book_demod[idx])
                else:
                    ones[ind].append(code_book_demod[idx])
        return zeros, ones

    def plot_const(self):
        const = self.code_book
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
        if self.M == 2:
            M = 'B'
        elif self.M == 4:
            M = 'Q'
        else:
            M = str(self.M)+"-"

        if self.gray_map == True:
            mapping = 'Gray'
        else:
            mapping = 'Binary'

        if self.bin_input == True:
            inputs = 'Binary'
        else:
            inputs = 'Decimal'

        plt.grid()
        plt.axvline(linewidth=1.0, color='black')
        plt.axhline(linewidth=1.0, color='black')
        plt.axis([-1.5,1.5,-1.5,1.5])
        plt.title(M+'PSK, phase rotation: '+str(round(self.phi, 5))+\
                  ', Mapping: '+mapping+', Input: '+inputs)
        plt.show()   
                                    
    def modulate(self, x):
        modulated = []
        if self.bin_input == True: 
            m = []
            n = int(np.log2(self.M))
            lenx = len(x)
            for c in range(int(lenx/n)):
                s = ''
                y = x[(c + (n - 1)*c):(((n - 1)*c) + (n - 1) + (1+c))]
                for d in y:
                    s = s+str(int(d))
                modulated.append(self.code_book[s])
        else:
            for a in x:
                modulated.append(self.code_book[a])
        return np.array(modulated)
     
    def demodulate(self, x):
        if self.decision_method == 'Approximate LLR':
            result = self.ApproxLLR(x, self.zeros, self.ones)
        elif self.decision_method == 'Exact LLR':
            result = self.ExactLLR(x, self.zeros, self.ones)
        elif self.decision_method == 'Hard':
            result = (np.sign(-self.ApproxLLR(x, self.zeros, self.ones)) + 1) / 2                        
        return result 
    
    
class QAMModem(Modem):
    def __init__(self, M, gray_map=True, bin_input=True):
        super().__init__(M, gray_map, bin_input)
        
        if np.sqrt(M) != np.fix(np.sqrt(M)) or np.log2(np.sqrt(M)) != np.fix(np.log2(np.sqrt(M))):
            raise ValueError('M must be a square of a power of 2')
        self.Type = 'QAM'
        c = np.sqrt(M)
        b = -2*(np.array(self.s) % c) + c - 1
        a = 2*np.floor(np.array(self.s)/c) - c + 1 
        self.m = list((a + 1j*b))
        self.code_book = self.__create_constellation(self.s, self.m)

    def modulate(self, x):
        modulated = []
        if self.bin_input== True:
            m = []
            n = int(np.log2(self.M))
            lenx = len(x)
            for c in range(int(lenx/n)):
                s=''
                y = x[(c+(n-1)*c):(((n-1)*c)+(n-1)+(1+c))]
                for d in y:
                    s = s+str(int(d))
                modulated.append(self.code_book[s])
        else:
            for a in x:
                modulated.append(self.code_book[a])
        return np.array(modulated)
