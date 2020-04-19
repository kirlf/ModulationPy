# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class Modem:
    def __init__(self, M, gray_map = True, bin_input = True, soft_decision = True, bin_output = True):
        
        N = np.log2(M) # bits per symbol
        if N != np.round(N):
            raise ValueError("M should be 2**n, with n=1, 2, 3...")
        if soft_decision == True and bin_output == False:
            raise ValueError("Non-binary output is available only for hard decision") 
        
        self.M = M # modulation order
        self.N = int(N) # bits per symbol
        self.m = [i for i in range(self.M)]    
        self.gray_map = gray_map
        self.bin_input = bin_input
        self.soft_decision = soft_decision
        self.bin_output = bin_output
        
    ''' SERVING METHODS '''

    def __gray_encoding(self, s):

        ''' Encodes the binary sequence by Gray encoding rule.         
        
        Parameters
        ----------
        s : list of ints
            Input binary sequence to be encoded by Gray.
        Returns
        -------
        s2: list of ints
            Output encoded by Gray sequence.
        '''

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

    def create_constellation(self, m, s):

        ''' Creates signal constellation.
        Parameters
        ----------
        m : list of ints
            Possible decimal values of the signal constellation (0 ... M-1).
        s : list of complex values
            Possible coordinates of the signal constellation.
        Returns
        -------
        dict_out: dict
            Output dictionary where 
            key is the bit sequence or decimal value and 
            value is the complex coordinate.         
        '''

        if self.bin_input == False and self.gray_map == False:
            dict_out = {k: v for k,v in zip(m, s)}    
        elif self.bin_input == False and self.gray_map == True:
            mg = self.__gray_encoding(m)
            dict_out = {k: v for k,v in zip(mg, s)}
        elif self.bin_input == True and self.gray_map == False:
            mb = self.de2bin(m)
            dict_out = {k: v for k,v in zip(mb, s)}
        elif self.bin_input == True and self.gray_map == True:
            mg = self.__gray_encoding(m)
            mgb = self.de2bin(mg)
            dict_out = {k: v for k,v in zip(mgb, s)}
        return dict_out

    def llr_preparation(self):


        ''' Creates the coordinates 
        where either zeros or ones can be placed in the signal constellation..
        Returns
        -------
        zeros : list of lists of complex values 
            The coordinates where zeros can be placed in the signal constellation.
        ones : list of lists of complex values 
            The coordinates where ones can be placed in the signal constellation.        
        '''

        code_book_demod = self.code_book
        
        zeros = [[] for i in range(self.N)]  
        ones = [[] for i in range(self.N)]

        b = self.de2bin(self.m)
        for idx, n in enumerate(b):
            for ind, m in enumerate(n):
                if self.bin_input == True:
                    if m == '0':
                        zeros[ind].append(code_book_demod[n])
                    else:
                        ones[ind].append(code_book_demod[n])
                else:
                    if m == '0':
                        zeros[ind].append(code_book_demod[idx])
                    else:
                        ones[ind].append(code_book_demod[idx])
        return zeros, ones

    ''' DEMODULATION ALGORITHMS '''
    
    def __ApproxLLR(self, x, noise_var):

        ''' Calculates approximate Log-likelihood Ratios (LLRs) [1].         
        
        Parameters
        ----------
        x : 1-D ndarray of complex values
            Received complex-valued symbols to be demodulated.
        noise_var: float
            Additive noise variance.
        Returns
        -------
        result: 1-D ndarray of floats
            Output LLRs.
        Reference:
            [1] Viterbi, A. J., "An Intuitive Justification and a 
                Simplified Implementation of the MAP Decoder for Convolutional Codes,"
                IEEE Journal on Selected Areas in Communications, 
                vol. 16, No. 2, pp 260–264, Feb. 1998
            
        '''
        
        zeros = self.zeros
        ones = self.ones
        LLR = []
        for (zero_i, one_i) in zip(zeros, ones):

            num = [((np.real(x) - np.real(z))**2)
                    + ((np.imag(x) - np.imag(z))**2)
                      for z in zero_i]
            denum = [(( np.real(x) - np.real(o))**2)
                    + ((np.imag(x) - np.imag(o))**2)
                      for o in one_i]
            
            num_post = np.amin(num, axis=0, keepdims=True)
            denum_post = np.amin(denum, axis=0, keepdims=True)

            llr = np.transpose(num_post[0]) - np.transpose(denum_post[0])
            LLR.append(-llr/noise_var)

        result = np.zeros((len(x)*len(zeros))) 
        for i, llr in enumerate(LLR):
            result[i::len(zeros)] = llr
        return result

    ''' METHODS TO EXECUTE '''
    
    def modulate(self, msg):
        ''' Modulates binary or decimal stream.
        Parameters
        ----------
        x : 1-D ndarray of ints
            Decimal or binary stream to be modulated.
        Returns
        -------
        modulated : 1-D array of complex values
            Modulated symbols (signal envelope).
        '''

        if (self.bin_input == True) and ((len(msg) % self.N) != 0):
        	raise ValueError("The length of the binary input should be a multiple of log2(M)")

        if (self.bin_input == True) and ((max(msg) > 1.) or (min(msg) < 0.)):
        	raise ValueError("The input values should be 0s or 1s only!")
        if (self.bin_input == False) and ((max(msg) > (self.M - 1)) or (min(msg) < 0.)):
        	raise ValueError("The input values should be in following range: [0, ... M-1]!")

        if self.bin_input == True: 
            msg = [str(bit) for bit in msg]
            splited = ["".join(msg[i:i + self.N]) 
                          for i in range(0, len(msg), self.N)] # subsequences of bits
            modulated = [self.code_book[s] for s in splited]
        else:
            modulated = [self.code_book[dec] for dec in msg]
        return np.array(modulated)
     
    def demodulate(self, x, noise_var=1.):
        ''' Demodulates complex symbols.
        Parameters
        ----------
        x : 1-D ndarray of complex symbols
            Decimal or binary stream to be demodulated.
        noise_var: float
            Additive noise variance.
        Returns
        -------
        result : 1-D array floats
            Demodulated message (LLRs or binary sequence).
        '''

        if self.soft_decision == True:
            result = self.__ApproxLLR(x, noise_var)
        else:
            if self.bin_output == True:
                llr = self.__ApproxLLR(x, noise_var)
                result = (np.sign(-llr) + 1) / 2 # NRZ-to-bin
            else:
                llr = self.__ApproxLLR(x, noise_var)
                result = self.bin2de((np.sign(-llr) + 1) / 2)                      
        return result 

    
class PSKModem(Modem):
    def __init__(self, M, phi=0, gray_map=True, bin_input=True, soft_decision=True, bin_output = True):
        super().__init__(M, gray_map, bin_input, soft_decision, bin_output)
        self.phi = phi 
        self.s = list(np.exp(1j*self.phi + 1j*2*np.pi*np.array(self.m)/self.M))
        self.code_book = self.create_constellation(self.m, self.s)
        self.zeros, self.ones = self.llr_preparation()  
      
    
    def de2bin(self, decs):
        ''' Converts values from decimal to binary representation.
        Parameters
        ----------
        decs : list of ints
            Input decimal values.
        Returns
        -------
        bin_out : list of ints
            Output binary sequences.
        '''
        if self.N % 2 == 0:
            bin_out = [np.binary_repr(d, width=self.N)[::-1] 
                  for d in decs]
        else:
            bin_out = [np.binary_repr(d, width=self.N) 
                  for d in decs]
        return bin_out

    def bin2de(self, bin_in):
        ''' Converts values from binary to decimal representation.
        Parameters
        ----------
        bin_in : list of ints
            Input binary values.
        Returns
        -------
        dec_out : list of ints
            Output decimal values.
        '''

        dec_out = []
        N = self.N # bits per modulation symbol (local variables are tiny bit faster)
        Ndecs = int(len(bin_in) / N) # length of the decimal output 
        for i in range(Ndecs):
            bin_seq = bin_in[i*N:i*N+N] # binary equivalent of the one decimal value 
            str_o = "".join([str(int(b)) for b in bin_seq]) # binary sequence to string
            if N % 2 == 0:
                str_o = str_o[::-1]
            dec_out.append(int(str_o, 2))
        return dec_out
    
    def plot_const(self):     
        ''' Plots signal constellation '''
        
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
                                  
                                  
class QAMModem(Modem):
    def __init__(self, M, gray_map=True, bin_input=True, soft_decision = True, bin_output = True):
        super().__init__(M, gray_map, bin_input, soft_decision, bin_output)
        
        if np.sqrt(M) != np.fix(np.sqrt(M)) or np.log2(np.sqrt(M)) != np.fix(np.log2(np.sqrt(M))):
            raise ValueError('M must be a square of a power of 2')

        self.m = [i for i in range(self.M)]  
        self.s = self.__qam_symbols()
        self.code_book = self.create_constellation(self.m, self.s)

        if self.gray_map:
            self.__gray_qam_arange()

        self.zeros, self.ones = self.llr_preparation()  


    def __qam_symbols(self):
        ''' Creates M-QAM complex symbols.'''        

        c = np.sqrt(self.M)
        b = -2*(np.array(self.m) % c) + c - 1
        a = 2*np.floor(np.array(self.m) / c) - c + 1 
        s = list((a + 1j*b))
        return  s

    def __gray_qam_arange(self):
        ''' Rearanges complex coordinates according to Gray coding requirements.
        '''   

        for idx, (key, item) in enumerate(self.code_book.items()):
            if (np.floor(idx / np.sqrt(self.M)) % 2) != 0:
                self.code_book[key] = np.conj(item)


    def de2bin(self, decs):
        ''' Converts values from decimal to binary representation.
        Parameters
        ----------
        decs : list of ints
            Input decimal values.
        Returns
        -------
        bin_out : list of ints
            Output binary sequences.
        '''
        bin_out = [np.binary_repr(d, width=self.N) for d in decs]
        return bin_out

    def bin2de(self, bin_in):
        ''' Converts values from binary to decimal representation.
        Parameters
        ----------
        bin_in : list of ints
            Input binary values.
        Returns
        -------
        dec_out : list of ints
            Output decimal values.
        '''

        dec_out = []
        N = self.N # bits per modulation symbol (local variables are tiny bit faster)
        Ndecs = int(len(bin_in) / N) # length of the decimal output 
        for i in range(Ndecs):
            bin_seq = bin_in[i*N:i*N+N] # binary equivalent of the one decimal value 
            str_o = "".join([str(int(b)) for b in bin_seq]) # binary sequence to string
            dec_out.append(int(str_o, 2))
        return dec_out 


    def plot_const(self):
        ''' Plots signal constellation '''

        if self.M <= 16:
            limits = np.log2(self.M)
            size = 'small'
        elif self.M == 64:
            limits = 1.5*np.log2(self.M)
            size = 'x-small'
        else:
            limits = 2.25*np.log2(self.M)
            size = 'xx-small'

        const = self.code_book
        fig = plt.figure(figsize=(6, 4), dpi=150)
        for i in list(const):
            x = np.real(const[i])
            y = np.imag(const[i])
            plt.plot(x, y, 'o', color='red')
            if x < 0:
                h = 'right'
                xadd = -.05
            else:
                h = 'left'
                xadd = .05
            if y < 0:
                v = 'top'
                yadd = -.05
            else:
                v = 'bottom'
                yadd = .05
            if (abs(x) < 1e-9 and abs(y) > 1e-9):
                h = 'center'
            elif abs(x) > 1e-9 and abs(y) < 1e-9:
                v = 'center'     
            plt.annotate(i,(x+xadd,y+yadd), ha=h, va=v, size=size)
        M = str(self.M)
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
        plt.axis([-limits,limits,-limits,limits])
        plt.title(M+'-QAM, Mapping: '+mapping+', Input: '+inputs)
        plt.show()   
