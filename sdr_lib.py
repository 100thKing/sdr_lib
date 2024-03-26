import numpy as np
import matplotlib.pyplot as plt
import random
import adi


def generate_binary_arr(len):
    arr = np.zeros(len)
    for i in range(len):
        arr[i] = random.randint(0,1)
    return arr

def string_to_bin(message):
    bin_result = ''.join(format(ord(x), '08b') for x in message)
    return bin_result


def bin_to_complex(binary):
    result = np.zeros(len(binary)/2)
    for i in range(0,len(binary),2):
        if binary[i] == 0 and binary[i+1] == 0:
            result[i] = (-1-1j)
        if binary[i] == 0 and binary[i+1] == 1:
            result[i] = (-1+1j)
        if binary[i] == 1 and binary[i+1] == 0:
            result[i] = (1-1j)
        if binary[i] == 1 and binary[i+1] == 1:
            result[i] = (+1+1j)
    return result

def modulate_qpsk(bit_array): 
    """5.1.3 QPSK modulation 
    """ 
    bit_array_Q = np.zeros(len(bit_array/2))
    bit_array_I = np.zeros(len(bit_array/2))
    complex_array = np.array([], dtype = complex) 
    for i in range(0,len(bit_array),2):
        bit_array_Q[i] = bit_array[i]
        bit_array_I[i] = bit_array[i+1]
    complex_array.resize( int(len(bit_array_I) ) )  
    # Modulation 
    for i in range(0, len(bit_array_I)): 
        real = 1 / np.sqrt(2) * (1 - 2 * bit_array_I[i]) 
        imag = 1 / np.sqrt(2) * (1 - 2 * bit_array_Q[i]) 
        complex_array[i] = complex(real, imag) 
    return complex_array 
 

def add_pylots(arr, pos, val):
    arr = np.insert(arr, 0, val)
    for i in range (0,len(arr),pos):
        arr = np.insert(arr, i, val)
    return arr

def repeat_elements(arr):
    repeated_arr = np.repeat(arr, 10)
    return repeated_arr

def add_barker(arr):
    barker = np.array([+1, +1, +1, -1, -1, +1, -1 ])
    signal = np.append(barker, arr)
    return signal

def higher_signal(signal):
    return signal * (2**14)


def signal_to_transmite(signal):
    signal_tx = add_barker(signal)
    signal_tx = repeat_elements(signal_tx)
    signal_tx =  higher_signal(signal_tx)
    return signal_tx

# TODO Сделать функцию configure_sdr для tx и rx +
def configure_sdr_rx(sample_rate=1e6, center_freq=2400e6, num_samps=10000, tx_cyclyc_buffer = False, ip = "ip:192.168.2.1"):
    sdr = adi.Pluto(ip)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_lo = int(center_freq)
    sdr.rx_buffer_size = num_samps
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 70.0
    #sdr.tx_lo = int(center_freq)
    #sdr.tx_cyclic_buffer = tx_cyclyc_buffer    
    return sdr

def configure_sdr_tx(sample_rate=1e6, center_freq=2400e6, num_samps=10000, tx_cyclyc_buffer = False, ip = "ip:192.168.2.1", tx_gain = -10, gain_mode = "manual"):
    # Config Tx  
    sdr = adi.Pluto(ip)
    sdr.tx_lo = int(center_freq)  
    sdr.tx_destroy_buffer()  
    sdr.tx_cyclic_buffer = tx_cyclyc_buffer 
    sdr.sample_rate = sample_rate  
    sdr.gain_control_mode_chan0 = gain_mode
    sdr.tx_hardwaregain_chan0 = tx_gain

    return sdr
