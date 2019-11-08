[![Build Status](https://travis-ci.com/kirlf/ModulationPy.svg?branch=master)](https://travis-ci.com/kirlf/ModulationPy)
[![PyPi](https://badge.fury.io/py/ModulationPy.svg)](https://pypi.org/project/ModulationPy/)
[![Coverage](https://coveralls.io/repos/kirlf/ModulationPy/badge.svg)](https://coveralls.io/r/kirlf/ModulationPy)

# ModulationPy

Digital baseband linear modems: M-PSK and M-QAM.

## Motivation

The main idea is to develop a Python module that allows replacing related to **baseband digital linear modulations** MatLab/Octave functions and objects.  This project is inspired by [CommPy](https://github.com/veeresht/CommPy) open-source project. 

## Theory basics

### 1. Linearity

Linear modulation schemes have a canonical form \[1\]:

<p align="center" style="text-align: center;"><img align="center" src="https://tex.s2cms.ru/svg/%20s(t)%20%3D%20s_I(t)cos(2%5Cpi%20f_c%20t)%20-%20s_Q(t)cos(2%5Cpi%20f_c%20t)%20%5Cqquad%20(1)" alt=" s(t) = s_I(t)cos(2\pi f_c t) - s_Q(t)cos(2\pi f_c t) \qquad (1)" /></p>

where <img src="https://tex.s2cms.ru/svg/s_I(t)" alt="s_I(t)" /> is the In-phase part, <img src="https://tex.s2cms.ru/svg/s_Q(t)" alt="s_Q(t)" /> is the Quadrature part, <img src="https://tex.s2cms.ru/svg/f_c" alt="f_c" /> is the carrier frequency, and <img src="https://tex.s2cms.ru/svg/t" alt="t" /> is the time moment. In-phase and Quadrature parts are low-pass signals that **linearly** correlate with an information signal. 

### 2. Baseband representation

Modulation scheme can also be modeled without consideration of the carrier frequency and bit duration. The baseband analogs can be used for research due to the main properties depend on the envelope (complex symbols).

### 3. Modulation order

Modulation order means number of possible modulation symbols. The number of bits per modulation symbol
depend on the modulation order:

<p align="center" style="text-align: center;"><img align="center" src="https://tex.s2cms.ru/svg/%20N%20%3D%20log_2(M)%20%5Cqquad(2)%20" alt=" N = log_2(M) \qquad(2) " /></p>

Modulation order relates to **gross bit rate** concept:

<p align="center" style="text-align: center;"><img align="center" src="https://tex.s2cms.ru/svg/%20R_b%20%3D%20R_slog_2(N)%20%5Cqquad(3)%20" alt=" R_b = R_slog_2(N) \qquad(3) " /></p>

where <img src="https://tex.s2cms.ru/svg/R_s" alt="R_s" /> is the baud or symbol rate. Baud rate usually relates to the coherence bandwidth <img src="https://tex.s2cms.ru/svg/B_c" alt="B_c" /> (see more in \[2\]).

> See more in ["Basics of linear digital modulations"](https://speakerdeck.com/kirlf/linear-digital-modulations) (slides).

## Installation

Released version on PyPi:

``` bash
$ pip install ModulationPy
```

To build by sources, clone from github and install as follows:

```bash
$ git clone https://github.com/kirlf/ModulationPy.git
$ cd ModulationPy
$ python3 setup.py install
```

## What are modems available?

- **M-PSK**: **P**hase **S**hift **K**eying
- **M-QAM**: **Q**uadratured **A**mplitude **M**odulation

where **M** is the modulation order.

### 1. M-PSK

M-PSK modem is available in ```class PSKModem``` with the following parameters:

| Parametr | Possible values | Description |
| ------------- |:-------------| :-----|
| ``` M ```      | positive integer values power of 2 | Modulation order. Specify the number of points in the signal constellation as scalar value that is a positive integer power of two.|
| ```phi``` | float values | Phase rotation. Specify the phase offset of the signal constellation, in radians, as a real scalar value. The default is 0.|
| ```gray_map``` | ```True``` or ```False``` | Specifies mapping rule. If parametr is ```True``` the modem works with Gray mapping, else it works with Binary mapping. The default is ```True```.|
| ```bin_input``` | ```True``` or ```False```| Specify whether the input of ```modulate()``` method is bits or integers. When you set this property to ```True```, the ```modulate()``` method input requires a column vector of bit values. The length of this vector must an integer multiple of log2(M). The default is ```True```.|
| ```soft_decision``` | ```True``` or ```False``` | Specify whether the output values of ```demodulate()``` method is demodulated as hard or soft decisions. If parametr is ```True``` the output will be Log-likelihood ratios (LLR's), else binary symbols. The default is ```True```.|
| ```bin_output``` | ```True``` or ```False```|Specify whether the output of ```demodulate()``` method is bits or integers. The default is ```True```.|

The mapping of into the modulation symbols is done by the [following formula](https://www.mathworks.com/help/comm/ref/comm.pskmodulator-system-object.html?s_tid=doc_ta):

<p align="center" style="text-align: center;"><img align="center" src="https://tex.s2cms.ru/svg/%20r%20%3D%20exp(j%5Cphi%20%2B%20j2%5Cpi%20m%2FM)" alt=" r = exp(j\phi + j2\pi m/M)" /></p>

where <img src="https://tex.s2cms.ru/svg/%20%5Cphi%20" alt=" \phi " /> is the phase rotation, <img src="https://tex.s2cms.ru/svg/%20m%20" alt=" m " /> is the decimal input symbol, and <img src="https://tex.s2cms.ru/svg/%20M%20" alt=" M " /> is the modulation order.

The input <img src="https://tex.s2cms.ru/svg/%20m%20" alt=" m " /> should be in range between *0* and *M-1*.

If the input is binary, the conversion from binary to decimal should be done before. Therefore, additional supportive method ``` __de2bin() ``` is implemented. This method has an additional heuristic: the bit sequence of "even" modulation schemes (e.g., QPSK) should be read right to left. 

### 2. M-QAM

M-QAM modem is available in ```class QAMModem``` with the following parameters:

| Parametr | Possible values | Description |
| ------------- |:-------------| :-----|
| ``` M ```      | positive integer values power of 2 | Modulation order. Specify the number of points in the signal constellation as scalar value that is a positive integer power of two.|
| ```gray_map``` | ```True``` or ```False``` | Specifies mapping rule. If parametr is ```True``` the modem works with Gray mapping, else it works with Binary mapping. The default is ```True```.|
| ```bin_input``` | ```True``` or ```False```| Specify whether the input of ```modulate()``` method is bits or integers. When you set this property to ```True```, the ```modulate()``` method input requires a column vector of bit values. The length of this vector must an integer multiple of log2(M). The default is ```True```.|
| ```soft_decision``` | ```True``` or ```False``` | Specify whether the output values of ```demodulate()``` method is demodulated as hard or soft decisions. If parametr is ```True``` the output will be Log-likelihood ratios (LLR's), else binary symbols. The default is ```True```.|
| ```bin_output``` | ```True``` or ```False```|Specify whether the output of ```demodulate()``` method is bits or integers. The default is ```True```.|

## How to use?

### 1. Initialization.

E.g., **QPSK** with the pi/4 phase offset, binary input and Gray mapping:  

```python
from ModulationPy import PSKModem, QAMModem
import numpy as np

modem = PSKModem(4, np.pi/4,
                 gray_map=True,
                 bin_input=True)
```

To show signal constellation use the ``` plot_const()``` method:

```python

modem.plot_const()

```

<img src="https://raw.githubusercontent.com/kirlf/ModulationPy/master/docs/img/qpsk_signconst.PNG" width="600" />


E.g. **16-QAM** with decimal input and Gray mapping

```python
modem = QAMModem(16,
                 gray_map=True, 
                 bin_input=False)

modem.plot_const()
```

<img src="https://raw.githubusercontent.com/kirlf/ModulationPy/master/docs/img/qam_signconst.PNG" width="600" />


### 2. Modulation and demodulation

To modulate and demodulate use ```modulate()``` and ```demodulate()``` methods.

The method ```modulate()``` has the one input argument: 

- decimal or binary stream to be modulated (```1-D ndarray of ints```).

The method ```demodulate()``` has the two input arguments: 

- data stream to be demodulated (```1-D ndarray of complex symbols```) and

- additive noise variance (```float```, default is 1.0).

E.g., QPSK (binary input/otput):

```python

import numpy as np
from ModulationPy import PSKModem

modem = PSKModem(4, np.pi/4, 
                 bin_input=True,
                 soft_decision=False,
                 bin_output=True)

msg = np.array([0, 0, 0, 1, 1, 0, 1, 1]) # input message

modulated = modem.modulate(msg) # modulation
demodulated = modem.demodulate(modulated) # demodulation

print("Modulated message:\n"+str(modulated))
print("Demodulated message:\n"+str(demodulated))

>>> Modulated message:
[ 0.70710678+0.70710678j  0.70710678-0.70710678j -0.70710678+0.70710678j
 -0.70710678-0.70710678j]
 
 >>> Demodulated message:
[0. 0. 0. 1. 1. 0. 1. 1.]
 
```

E.g., QPSK (decimal input/otput):

``` python

import numpy as np
from ModulationPy import PSKModem

modem = PSKModem(4, np.pi/4, 
                 bin_input=False,
                 soft_decision=False,
                 bin_output=False)

msg = np.array([0, 1, 2, 3]) # input message

modulated = modem.modulate(msg) # modulation
demodulated = modem.demodulate(modulated) # demodulation

print("Modulated message:\n"+str(modulated))
print("Demodulated message:\n"+str(demodulated))

>>> Modulated message:
[ 0.70710678+0.70710678j -0.70710678+0.70710678j  0.70710678-0.70710678j
 -0.70710678-0.70710678j]
 
 >>> Demodulated message:
[0, 1, 2, 3]

```

E.g., 16-QAM (decimal input/output):

``` python

import numpy as np
from ModulationPy import QAMModem

modem = PSKModem(16, 
                 bin_input=False,
                 soft_decision=False,
                 bin_output=False)

msg = np.array([i for i in range(16)]) # input message

modulated = modem.modulate(msg) # modulation
demodulated = modem.demodulate(modulated) # demodulation

print("Demodulated message:\n"+str(demodulated))

>>> Demodulated message:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

```


### 3. Bit-error ratio performance

Let us demonstrate this at example with the following system model:

![](https://raw.githubusercontent.com/kirlf/ModulationPy/master/docs/img/simulator_scheme.PNG)

*Fig. 1. The structural scheme of the test communication model.*

The simulation results will be compared with theoretical curves \[3\]:

<p align="center" style="text-align: center;"><img align="center" src="https://tex.s2cms.ru/svg/%20%0AP_b%20%3D%20%5Cfrac%7Berfc%20%5Cleft(%20%5Csqrt%7Blog_2(M)%5Cfrac%7BE_b%7D%7BN_o%7D%7D%5Csin%5Cleft(%20%5Cfrac%7B%5Cpi%7D%7BM%7D%20%5Cright)%20%5Cright)%7D%7Blog_2(M)%7D%20%5Cqquad%20(4)%0A" alt=" 
P_b = \frac{erfc \left( \sqrt{log_2(M)\frac{E_b}{N_o}}\sin\left( \frac{\pi}{M} \right) \right)}{log_2(M)}
" /></p>

where <img src="https://tex.s2cms.ru/svg/M%20%3E4%20" alt="M &gt;4 " />, <img src="https://tex.s2cms.ru/svg/erfc(*)" alt="erfc(*)" /> denotes the [complementary error function](https://en.wikipedia.org/wiki/Error_function#Complementary_error_function), <img src="https://tex.s2cms.ru/svg/E_b" alt="E_b" /> is the energy per one bit, and <img src="https://tex.s2cms.ru/svg/N_o" alt="N_o" /> is the noise spectral density.

In case of BPSK and QPSK the following formula should be used for error probability:

<p align="center" style="text-align: center;"><img align="center" src="https://tex.s2cms.ru/svg/%20%0AP_%7Bb%2C%20BQ%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7Derfc%20%5Cleft(%20%5Csqrt%7B%5Cfrac%7BE_b%7D%7BN_o%7D%7D%5Cright)%7D%20%5Cqquad%20(5)%0A" alt=" 
P_{b, BQ} = \frac{1}{2}erfc \left( \sqrt{\frac{E_b}{N_o}}\right)} \qquad (5)
" /></p>

The source code of the simulation is presented bellow:

``` python

import numpy as np
from ModulationPy import PSKModem

def BER_calc(a, b):
    num_ber = np.sum(np.abs(a - b))
    ber = np.mean(np.abs(a - b))
    return int(num_ber), ber

def BER_psk(M, EbNo):
    EbNo_lin = 10**(EbNo/10)
    if M > 4:
        P = special.erfc(np.sqrt(EbNo_lin*np.log2(M))*np.sin(np.pi/M)) 
            / np.log2(M)
    else:
        P = 0.5*special.erfc(np.sqrt(EbNo_lin))
    return P
    
EbNos = np.array([i for i in range(30)]) # array of Eb/No in dBs 
N = 100000 # number of symbols per the frame
N_c = 100 # number of trials

Ms = [4, 8, 16, 32] # modulation orders

''' Simulation loops '''

mean_BER = np.empty((len(EbNos), len(Ms)))
for idxM, M in enumerate(Ms):
    print(M)
    BER = np.empty((N_c,))
    k = np.log2(M) #number of bit per modulation symbol

    modem = PSKModem(M,
                 gray_map=True,
                 bin_input=True,
                 soft_decision=False,
                 bin_output=True)

    for idxEbNo, EbNo in enumerate(EbNos):
        print(EbNo)
        snrdB = EbNo + 10*np.log10(k) # Signal-to-Noise ratio (in dB)
        noiseVar = 10**(-snrdB/10) # noise variance (power)

        for cntr in range(N_c):
            message_bits = np.random.randint(0, 2, int(N*k)) # message
            modulated = modem.modulate(message_bits) # modulation

            Es = np.mean(np.abs(modulated)**2) # symbol energy
            No = Es/((10**(EbNo/10))*np.log2(M)) # noise spectrum density

            noisy = modulated + np.sqrt(No/2)*\
              (np.random.randn(modulated.shape[0])+\
              1j*np.random.randn(modulated.shape[0])) # AWGN

            demodulated = modem.demodulate(noisy, noise_var=noiseVar)
            NumErr, BER[cntr] = BER_calc(message_bits, 
                                            demodulated) # bit-error ratio

    mean_BER[idxEbNo, idxM] = np.mean(BER) # averaged bit-error ratio


''' Theoretical results '''

BER_theor = np.empty((len(EbNos), len(Ms)))
for idxM, M in enumerate(Ms):
    BER_theor[:, idxM] = BER_psk(M, EbNos)


''' Curves '''

fig, ax = plt.subplots(figsize=(10,7), dpi=300)

plt.semilogy(EbNos, BER_theor[:,0], 'g-', label = 'QPSK (theory)')
plt.semilogy(EbNos, BER_theor[:,1], 'b-', label = '8-PSK (theory)')
plt.semilogy(EbNos, BER_theor[:,2], 'k-', label = '16-PSK (theory)')
plt.semilogy(EbNos, BER_theor[:,3], 'r-', label = '32-PSK (theory)')

plt.semilogy(EbNos, mean_BER[:,0], 'g-o', label = 'QPSK (simulation)')
plt.semilogy(EbNos, mean_BER[:,1], 'b-o', label = '8-PSK (simulation)')
plt.semilogy(EbNos, mean_BER[:,2], 'k-o', label = '16-PSK (simulation)')
plt.semilogy(EbNos, mean_BER[:,3], 'r-o', label = '32-PSK (simulation)')

ax.set_ylim(1e-7, 2)
ax.set_xlim(0, 25.1)

plt.title("M-PSK")
plt.xlabel('EbNo (dB)')
plt.ylabel('BER')
plt.grid()
plt.legend(loc='upper right')
plt.savefig('psk_ber.png')

```

<img src="https://raw.githubusercontent.com/kirlf/ModulationPy/master/docs/img/psk_ber.png" width="750" />

It works.  Well done.



## References

1. Haykin S. Communication systems. – John Wiley & Sons, 2008. — p. 93 
2. Goldsmith A. Wireless communications. – Cambridge university press, 2005. – p. 88-92
3. Link Budget Analysis: Digital Modulation, Part 3 www.AtlantaRF.com
