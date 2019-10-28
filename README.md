[![Build Status](https://travis-ci.com/kirlf/ModulationPy.svg?branch=master)](https://travis-ci.com/kirlf/ModulationPy)
[![PyPi](https://badge.fury.io/py/ModulationPy.svg)](https://pypi.org/project/ModulationPy/)
[![Coverage](https://coveralls.io/repos/kirlf/ModulationPy/badge.svg)](https://coveralls.io/r/kirlf/ModulationPy)

# ModulationPy

Digital linear modems: M-PSK and M-QAM.

## Motivation

The main idea is to develop a Python module that allows replacing related to **digital linear modulations** MatLab/Octave functions and objects.  This project is inspired by [CommPy](https://github.com/veeresht/CommPy) open-source project. 

## Theory basics
  
See: [Basics of linear digital modulations](https://speakerdeck.com/kirlf/linear-digital-modulations) (slides).

## Installation

Released version on PyPi:

``` bash
$ pip install ModulationPy==0.1.4
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

1) To show signal constellation use ``` plot_const()``` method:

``` python
from ModulationPy import PSKModem, QAMModem
import numpy as np

PSKModem(4, np.pi/4,  gray_map=True, bin_input=True, soft_decision = False, bin_output = True).plot_const()
QAMModem(16, gray_map=True, bin_input=False, soft_decision = False, bin_output = False).plot_const()

```

<img src="https://raw.githubusercontent.com/kirlf/ModulationPy/master/docs/img/qpsk_signconst.PNG" width="600" />
<img src="https://raw.githubusercontent.com/kirlf/ModulationPy/master/docs/img/qam_signconst.PNG" width="600" />

2. To modulate and demodulate use ```modulate()``` and ```demodulate()``` methods.

[EXAMPLE in progress]
