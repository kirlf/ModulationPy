# Defining Methods

## Supportive functions

Gray encoder: encode some input binary sequence by Gray mapping rule.

```py
def GRAYencoder(x):
    for idx in range(len(x)):
        if idx == 0:
            y = x[idx]
        else:
            y = y + str ( int( x[idx] ) ^ int( x[idx-1] ) )
    return y
```

### M-PSK modulator initialization

### Inputs:

* **modulation\_order \(int\)**:  Modulation Order \(usually, noted as M\);

* **phase\_shift \(float\)**: Phase Shift \(usually, noted as phi\). Default is pi/4;

* **symbol\_mapping \(string\)**:  Mapping rule that will be used. 'Gray' or 'Binary' . Default is 'Gray';

* **input\_type \(string\)**: Type of the input that will be used. 'Binary' \(0 or 1\) or 'Decimal' \(0, 1, ... M-1\). Default is 'Binary'.

### Outputs:

* **dict\_out \(dict\)**: Dictionary with {"combination of bits or decimal value": complex value according to the signal constellation} structure.

```py
def PSKmodulator(modulation_order, phase_shift = np.pi/4, symbol_mapping = 'Gray', input_type = 'Binary'):
    s = [0+i for i in range(modulation_order)]
    m = list(np.exp(1j*phase_shift + 1j*2*np.pi*np.array(s)/modulation_order))
    dict_out = {}
    if input_type != 'Binary' and input_type != 'Decimal':
        print("Wrong input data type (should be 'Decimal' or 'Binary'). Now input_type = " + str(input_type))
        sys.exit(0) 
    elif input_type == 'Decimal':
        if symbol_mapping != 'Binary' and symbol_mapping != 'Gray':
            print("Wrong mapping rule (should be 'Gray' or 'Binary'). Now symbol_mapping = " + str(symbol_mapping))
            sys.exit(0)
        elif symbol_mapping == 'Gray':
            s2 = []
            for i in s:
                symbol = bin(i)[2:]
                if len(symbol) < np.log2(modulation_order):
                    symbol = int( (np.log2(modulation_order) - len(symbol)) )*'0'+symbol
                s2.append( int(GRAYencoder(symbol), 2 ) )
            s = s2
    elif input_type == 'Binary':
        if symbol_mapping == 'Binary':
            b = []
            for i in s:
                a = bin(i)[2:]
                if len(a) < np.log2(modulation_order):
                    a = int((np.log2(modulation_order) - len(a)))*'0'+a
                if np.log2(modulation_order)%2 == 0:
                    a = a[::-1]
                b.append(a)
            s = b
        elif symbol_mapping == 'Gray':
            s2 = []
            for i in s:
                symbol = bin(i)[2:]
                if len(symbol) < np.log2(modulation_order):
                    symbol = int( (np.log2(modulation_order) - len(symbol)) )*'0'+symbol
                s2.append(GRAYencoder(symbol))
            s = []
            for i in s2:
                if np.log2(modulation_order)%2 == 0:
                    i = i[::-1]
                s.append(i)
        else:
            print("Wrong mapping rule (should be 'Gray' or 'Binary'). Now input_type = " + str(symbol_mapping))
            sys.exit(0)
    for x, y in zip(s, m):
        dict_out[x] = y
    return dict_out
```

## M-PSK demodulator initialization

### Inputs:

* **modulation\_order \(int\):**  Modulation Order \(usually, noted as M\);

* **phase\_shift \(float\):** Phase Shift \(usually, noted as phi\). Default is pi/4.

* **symbol\_mapping \(string\):**  Mapping rule that will be used. 'Gray' or 'Binary' . Default is 'Gray'.

### Otputs:

* **zeros \(list\):** List of complex values that consists modulation symbols with zero bits;

* **ones \(list\):**  List of complex values that consists modulation symbols with one bits.

### Required function:

* PSKModulator\(\) 

```py
def PSKDemodulator(modulation_order, phase_shift = np.pi/4, symbol_mapping = 'Gray'):
    zeros = []
    ones = []
    for c in range(int(np.log2(modulation_order))):
        zeros.append([])
        ones.append([])
    codebook = PSKmodulator(modulation_order, phase_shift, symbol_mapping, 'Decimal')
    s = [i for i in range(modulation_order)]
    b = []
    for i in s:
        a = bin(i)[2:]
        if len(a) < np.log2(modulation_order):
            a = int((np.log2(modulation_order) - len(a)))*'0'+a
        if np.log2(modulation_order)%2 == 0:
            a = a[::-1]
        b.append(a)
    for idx, n in enumerate(b):
        for ind, m in enumerate(n):
            if m == '0':
                zeros[ind].append(codebook[idx])
            else:
                ones[ind].append(codebook[idx])
    return zeros, ones
```



