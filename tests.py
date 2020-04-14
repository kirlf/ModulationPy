import unittest
import numpy as np
from ModulationPy.ModulationPy import QAMModem, PSKModem


class TestThing(unittest.TestCase):
  def test_QAMModem(self):
    """ QAM modem  modulation and demodulation tests"""
    M = [4, 16, 64] # modulation order
    gm = [True, False] # Gray mapping
    bn = [True, False] # binary input

    for m in M:
        for g in gm:
            for b in bn:
                if b == True:
                    mlp = int(np.log2(m))
                    sz = mlp*int(1e4)
                else:
                    sz = int(1e5)
                msg = np.random.randint(2, size=sz)
                Modem = QAMModem(m, gray_map=g, bin_input=b, soft_decision = False, bin_output = b)
                
                np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

  def test_PSKModem(self):
    """ PSK modem  modulation and demodulation tests"""
    M = [2, 4, 8, 16, 32] # modulation order
    gm = [True, False] # Gray mapping
    bn = [True, False] # binary input

    for m in M:
        for g in gm:
            for b in bn:
                if b == True:
                    mlp = int(np.log2(m))
                    sz = mlp*int(1e4)
                else:
                    sz = int(1e5)
                msg = np.random.randint(2, size=sz)
                Modem = PSKModem(m, gray_map=g, bin_input=b, soft_decision = False, bin_output = b)
                np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

if __name__ == '__main__':
    unittest.main()

