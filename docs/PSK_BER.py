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
