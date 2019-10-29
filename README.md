[![Build Status](https://travis-ci.com/kirlf/ModulationPy.svg?branch=master)](https://travis-ci.com/kirlf/ModulationPy)
[![PyPi](https://badge.fury.io/py/ModulationPy.svg)](https://pypi.org/project/ModulationPy/)
[![Coverage](https://coveralls.io/repos/kirlf/ModulationPy/badge.svg)](https://coveralls.io/r/kirlf/ModulationPy)

# ModulationPy

Digital linear modems: M-PSK and M-QAM.

## Motivation

The main idea is to develop a Python module that allows replacing related to **digital linear modulations** MatLab/Octave functions and objects.  This project is inspired by [CommPy](https://github.com/veeresht/CommPy) open-source project. 

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

To build by sources, clone from github and install as follows::

```bash
$ git clone https://github.com/kirlf/ModulationPy.git
$ cd ModulationPy
$ python3 setup.py install
```

## What are modems available?

- **M-PSK**: **P**hase **S**hift **K**eying
- **M-QAM**: **Q**uadratured **A**mplitude **M**odulation

where **M** is the modulation order.

M-PSK modem is available in ```class PSKModem``` with the following parameters:

| Parametr | Possible values | Description |
| ------------- |:-------------| :-----|
| ``` M ```      | positive integer values power of 2 | Modulation order. Specify the number of points in the signal constellation as scalar value that is a positive integer power of two.|
| ```phi``` | float values | Phase rotation. Specify the phase offset of the signal constellation, in radians, as a real scalar value. The default is 0.|
| ```gray_map``` | ```True``` or ```False``` | Specifies mapping rule. If parametr is ```True``` the modem works with Gray mapping, else it works with Binary mapping. The default is ```True```.|
| ```bin_input``` | ```True``` or ```False```| Specify whether the input of ```modulate()``` method is bits or integers. When you set this property to ```True```, the ```modulate()``` method input requires a column vector of bit values. The length of this vector must an integer multiple of log2(M). The default is ```True```.|
| ```soft_decision``` | ```True``` or ```False``` | Specify whether the output values of ```demodulate()``` method is demodulated as hard or soft decisions. If parametr is ```True``` the output will be Log-likelihood ratios (LLR's), else binary symbols. The default is ```True```.|
| ```bin_output``` | ```True``` or ```False```|Specify whether the output of ```demodulate()``` method is bits or integers. The default is ```True```.|

M-QAM modem is available in ```class QAMModem``` with the following parameters:

| Parametr | Possible values | Description |
| ------------- |:-------------| :-----|
| ``` M ```      | positive integer values power of 2 | Modulation order. Specify the number of points in the signal constellation as scalar value that is a positive integer power of two.|
| ```gray_map``` | ```True``` or ```False``` | Specifies mapping rule. If parametr is ```True``` the modem works with Gray mapping, else it works with Binary mapping. The default is ```True```.|
| ```bin_input``` | ```True``` or ```False```| Specify whether the input of ```modulate()``` method is bits or integers. When you set this property to ```True```, the ```modulate()``` method input requires a column vector of bit values. The length of this vector must an integer multiple of log2(M). The default is ```True```.|
| ```soft_decision``` | ```True``` or ```False``` | Specify whether the output values of ```demodulate()``` method is demodulated as hard or soft decisions. If parametr is ```True``` the output will be Log-likelihood ratios (LLR's), else binary symbols. The default is ```True```.|
| ```bin_output``` | ```True``` or ```False```|Specify whether the output of ```demodulate()``` method is bits or integers. The default is ```True```.|

## How to use?

### 1. Initialazation.

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



2. To modulate and demodulate use ```modulate()``` and ```demodulate()``` methods.

[EXAMPLE in progress]


## References

1. Haykin S. Communication systems. – John Wiley & Sons, 2008. — p. 93 
2. Goldsmith A. Wireless communications. – Cambridge university press, 2005. – p. 88-92
