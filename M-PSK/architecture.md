# M-PSK modem

To avoid large computations in case of the iterative simulation the structure is devided to two main plains:

* Initialization of the modulator;
* Modulation \(see Shared blocks\).

## Initialization of the modulator: explanation

The main goal of this part is to construct dictionary where keys are combination of the bits or decimal values and values are complex values obtained by the following formula:


$$
\mathbf{m} = e^{j\phi+\frac{j2\pi\mathbf{d}}{M}}
$$


where m is the vector of complex values \(modulation symbols\), phi is the phase shift, M is the modulation order, d is the vector of the decimal values in range from 0 to M-1.

![](/assets/import.png)

Fig.1. Signal constellation for pi/4 QPSK.



![](/assets/import2.png)

Fig. 2. Block scheme of the work of the M-PSK modulator initialization module.

## Initialization of the demodulator : explanation

The main idea of this block is to construct two lists \(zeros and ones\) where index \(+1 since indeces start from 0 in Python\) means possible position of certain bit \(0 in zeros or 1 in ones\)  and value means possible modulation symbol \(complete\) where this bit can be.

Examples \(pi/4 QPSK\) :

* bit 0 can belong 0.7+0.7i or 0.7-0.7i if we consider the first position.

* bit 1 can belong -0.7-0.7i or 0.7-0.7i if we consider the second position

etc.

