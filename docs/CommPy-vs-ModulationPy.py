import numpy as np
import commpy.modulation as modulation
from ModulationPy import PSKModem
import time

N = int(1e4) # frame length
Nc = int(1e4) # number of trials

modem = PSKModem(4) # our class initialization
t = [] # for modulation time averaging
td = [] # for demudulation time averaging

msg = np.random.randint(0, 2, N)
for counter in range(Nc):
    t1 = time.time() # tic
    m = modem.modulate(msg) #modulation
    t2 = time.time() # toc
    t.append(t2 -t1)
    t1 = time.time() # tic
    modem.demodulate(m) # demodulation
    t2 = time.time() #toc
    td.append(t2 -t1)


res_m = np.mean(np.array(t))
res_dm = np.mean(np.array(td))

print("ModulationPy:")
print("\n - modulation:")
print(str(np.round(res_m*1e3, 2))+' ms')
print("\n - demodulation:")
print(str(np.round(res_dm*1e3, 2))+' ms')


modem = modulation.PSKModem(4) # CommPy class initialization
t = [] # for modulation time averaging
td = [] # for demudulation time averaging

msg = np.random.randint(0, 2, N)
for counter in range(Nc):
    t1 = time.time() # tic
    m = modem.modulate(msg) #modulation
    t2 = time.time() # toc
    t.append(t2 -t1)
    t1 = time.time() # tic
    modem.demodulate(m, demod_type='soft') # demodulation
    t2 = time.time() #toc
    td.append(t2 -t1)

res_m = np.mean(np.array(t))
res_dm = np.mean(np.array(td))

print("scikit-commpy:")
print("\n - modulation:")
print(str(np.round(res_m*1e3, 2))+' ms')
print("\n - demodulation:")
print(str(np.round(res_dm*1e3, 2))+' ms')
