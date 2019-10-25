from ModulationPy.ModulationPy import *

M = [4, 16, 64, 256, 1024]
gm = [True, False]
bn = [True, False]

for m in M:
    print(m)
    for g in gm:
        print('Gray mapping: '+str(g))
        for b in bn:
            print('Bin input/output: '+str(b))
            if b == True:
                mlp = int(np.log2(m))
                sz = mlp*int(1e4)
            else:
                sz = int(1e5)
            msg = np.random.randint(2, size=sz)
            Modem = QAMModem(m, gray_map=g, bin_input=b, soft_decision = False, bin_output = b)
            if np.array_equal(msg, Modem.demodulate(Modem.modulate(msg))) != True:
                raise ValueError("Test failed")
            else:
                print('Passed.')

M = [2, 4, 8, 16, 32]
gm = [True, False]
bn = [True, False]

for m in M:
    print(m)
    for g in gm:
        print('Gray mapping: '+str(g))
        for b in bn:
            print('Bin input/output: '+str(b))
            if b == True:
                mlp = int(np.log2(m))
                sz = mlp*int(1e4)
            else:
                sz = int(1e5)
            msg = np.random.randint(2, size=sz)
            Modem = PSKModem(m, gray_map=g, bin_input=b, soft_decision = False, bin_output = b)
            if np.array_equal(msg, Modem.demodulate(Modem.modulate(msg))) != True:
                raise ValueError("Test failed")
            else:
                print('Passed.')
