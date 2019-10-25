[![Build Status](https://travis-ci.com/kirlf/ModulationPy.svg?branch=master)](https://travis-ci.com/kirlf/ModulationPy)



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

## How to use?

1) To show signal constellation use ``` plot_const()``` method:

``` python
from ModulationPy import PSKModem, QAMModem
import numpy as np

PSKModem(4, np.pi/4,  gray_map=True, bin_input=True, soft_decision = False, bin_output = True).plot_const()
QAMModem(16, gray_map=True, bin_input=False, soft_decision = False, bin_output = False).plot_const()

```

<img src="https://raw.githubusercontent.com/kirlf/ModulationPy/master/doc/img/qpsk_signconst.PNG" width="600" />
<img src="https://raw.githubusercontent.com/kirlf/ModulationPy/master/doc/img/qam_signconst.PNG" width="600" />

2. To modulate and demodulate use ```modulate()``` and ```demodulate()``` methods.

[EXAMPLE in progress]
