import unittest
import numpy as np
from ModulationPy.ModulationPy import QAMModem, PSKModem


class TestThing(unittest.TestCase):
  def test_QAMModem(self):
    """ QAM modem  modulation and demodulation tests"""
    Ms = [4, 16, 64] # modulation orders

    for M in Ms:

      print("Modulation order: "+str(M))

      # Hard decision, Gray mapping, binary IO
      mlpr = int(np.log2(M)) #multiplier
      size = mlpr*int(1e4) 
      msg = np.random.randint(2, size=size)
      
      Modem = QAMModem(M, gray_map=True, bin_input=True, soft_decision = False, bin_output = True)
      np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

      # Hard decision, binary mapping, binary IO
      Modem = QAMModem(M, gray_map=False, bin_input=True, soft_decision = False, bin_output = True)
      np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

      # Hard decision, Gray mapping, non-binary IO
      size = int(1e5)
      msg = np.random.randint(2, size=size)
      Modem = QAMModem(M, gray_map=True, bin_input=False, soft_decision = False, bin_output = False)
      np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

      # Hard decision, binary mapping, non-binary IO
      Modem = QAMModem(M, gray_map=False, bin_input=False, soft_decision = False, bin_output = False)
      np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))


  def test_PSKModem(self):
    """ PSK modem  modulation and demodulation tests"""
    Ms = [2, 4, 8, 16, 32] # modulation order
    for M in Ms:

      print("Modulation order: "+str(M))

      # Hard decision, Gray mapping, binary IO
      mlpr = int(np.log2(M)) #multiplier
      size = mlpr*int(1e4) 
      msg = np.random.randint(2, size=size)
      
      Modem = PSKModem(M, gray_map=True, bin_input=True, soft_decision = False, bin_output = True)
      np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

      # Hard decision, binary mapping, binary IO
      Modem = PSKModem(M, gray_map=False, bin_input=True, soft_decision = False, bin_output = True)
      np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

      # Hard decision, Gray mapping, non-binary IO
      size = int(1e5)
      msg = np.random.randint(2, size=size)
      Modem = PSKModem(M, gray_map=True, bin_input=False, soft_decision = False, bin_output = False)
      np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

      # Hard decision, binary mapping, non-binary IO
      Modem = PSKModem(M, gray_map=False, bin_input=False, soft_decision = False, bin_output = False)
      np.testing.assert_array_equal(msg, Modem.demodulate(Modem.modulate(msg)))

if __name__ == '__main__':
    unittest.main()

