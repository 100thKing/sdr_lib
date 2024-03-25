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
 

def Add_Pylots(arr, pos, val):
    arr = np.insert(arr, 0, val)
    for i in range (0,len(arr),pos):
        arr = np.insert(arr, i, val)
    return arr

def repeat_elements(arr):
    repeated_arr = np.repeat(arr, 10)
    return repeated_arr

def Add_Barker(arr):
    barker = np.array([+1, +1, +1, -1, -1, +1, -1 ])
    signal = np.append(barker, arr)
    return signal

def higher_signal(signal):
    return signal * (2**14)


def signal_to_transmite(signal):
    signal_tx = Add_Barker(signal)
    signal_tx = repeat_elements(signal_tx)
    signal_tx =  higher_signal(signal_tx)
    return signal_tx

# TODO Сделать функцию configure_sdr для tx и rx +
def configure_sdr_rx(sample_rate=1e6, center_freq=2400e6, num_samps=10000, tx_cyclyc_buffer = False, ip = "ip:192.168.3.1"):
    sdr = adi.Pluto(ip)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_lo = int(center_freq)
    sdr.rx_buffer_size = num_samps
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 70.0
    #sdr.tx_lo = int(center_freq)
    #sdr.tx_cyclic_buffer = tx_cyclyc_buffer    
    return sdr

def configure_sdr_tx(sample_rate=1e6, center_freq=2400e6, num_samps=10000, tx_cyclyc_buffer = False, ip = "ip:192.168.3.1", tx_gain = -10, gain_mode = "manual"):
    # Config Tx  
    sdr = adi.Pluto(ip)
    sdr.tx_lo = int(center_freq)  
    sdr.tx_destroy_buffer()  
    sdr.tx_cyclic_buffer = tx_cyclyc_buffer 
    sdr.sample_rate = sample_rate  
    sdr.gain_control_mode_chan0 = gain_mode
    sdr.tx_hardwaregain_chan0 = tx_gain

    return sdr


# Пример использования функции:
# sdr_rx = configure_sdr(sample_rate=2e6, center_freq=2500e6, num_samps=5000)


def sdr_settings():
    # Попробуем отправить просто 1 синусоиду и принять ее же 
    sample_rate = 1e6 # Hz ширина полосы 
    center_freq = 2400e6# Hz 
    num_samps = 10000 # number of samples per call to rx() 
    
    sdr = adi.Pluto("ip:192.168.3.1") 
    sdr.sample_rate = int(sample_rate)     
    
    # Config Rx 
    sdr.rx_lo = int(center_freq) 
    sdr.rx_buffer_size = num_samps 
    sdr.gain_control_mode_chan0 = 'manual' #fast_attack, slow_attack 
    sdr.rx_hardwaregain_chan0 = 70.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC 
    sdr.tx_lo = int(center_freq) 
    sdr.tx_cyclic_buffer = False 

# def QPSK_coordinates(complex):
#     I_b = complex[0::2]
#     Q_b = complex[1::2]
#     QPSK_symb = []

#     for i in range(0, len(I_b)):
#         if I_b[i] == -1 and Q_b[i] == -1:
#             QPSK_symb.append(1/np.sqrt(2) + 1j/np.sqrt(2))
#         elif I_b[i] == -1 and Q_b[i] == 1:
#             QPSK_symb.append(-1/np.sqrt(2) + 1j/np.sqrt(2))
#         elif I_b[i] == 1 and Q_b[i] == -1:
#             QPSK_symb.append(1/np.sqrt(2) - 1j/np.sqrt(2))
#         elif I_b[i] == 1 and Q_b[i] == 1:
#             QPSK_symb.append(-1/np.sqrt(2) - 1j/np.sqrt(2))
    
#     signal = np.array(QPSK_symb)
#     return signal

# def complex(a):
#     if (a[0] == (-1-1j)):
#         return 0
#     if (a[0] == (-1+1j)):
#         return 1
#     if (a[0] == (1-1j)):
#         return 2
#     if (a[0] == (1+1j)):
#         return 3

# def QPSK_modulation(complex_arr):
#     buf = np.array([])
#     complex_message = np.empty(1,1)
#     for i in range(0,len(complex_arr)):
#         buf = np.append(buf,complex_arr[i])
#         if (i % 2 == 1) and (i!=0):
#             complex_message = np.append(complex_message,complex(buf))
#             # print(buf, buf[0] == '1')
#             buf = np.array([ ])
#     x_degrees = complex_message*360/4.0 +45 
#     x_radians = x_degrees*np.pi/180.0 # sin() и cos() в рад.
#     x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians)
#     samples = np.repeat(x_symbols, 10)
#     return samples




# def QPSK_demodulation(complex_array):
#     binary_array = np.zeros(len(complex_array)*2)
#     for i in range(len(complex_array)):
#         real_part = complex_array[i].real
#         imag_part = complex_array[i].imag
#         if real_part == -1 and imag_part == -1:
#             binary_array[i*2] = 0
#             binary_array[i*2 + 1] = 0
#         elif real_part == -1 and imag_part == 1:
#             binary_array[i*2] = 0
#             binary_array[i*2 + 1] = 1
#         elif real_part == 1 and imag_part == -1:
#             binary_array[i*2] = 1
#             binary_array[i*2 + 1] = 0
#         elif real_part == 1 and imag_part == 1:
#             binary_array[i*2] = 1
#             binary_array[i*2 + 1] = 1
#     return binary_array


# def generate_complex_numbers(n):
#     complex_numbers = []
#     for i in range(n):
#         real_part = random.uniform(-2, 2)
#         imag_part = random.uniform(-2, 2)
#         # print(real_part,' ', imag_part)
#         if (real_part > 0):
#             real_part = 1
#         else:
#             real_part = -1
#         if (imag_part > 0):
#             imag_part = 1
#         else:
#             imag_part = -1
#         complex_num = real_part +  imag_part * 1j
#         complex_numbers.append(complex_num)
#     return complex_numbers




# def signal_to_transmite():
    T=1e-4 # Длительность символа
    Nc=16 # Количество поднесущих
    df =1/T # Частотный интервал между поднесущими
    ts=T/Nc # Интервал дискретизации
    k=1
    t=ts * np.arange( 0 , Nc )
    s =1/np.sqrt (T) *np.exp (1j *2*np.pi *k* df * t ) # Формированиеодной поднесущей с частотой f ∗ d f
    sc_matr = np.zeros ( ( Nc , len ( t ) ) , dtype=complex )
    sd = np.zeros ( ( 1 , Nc ) , dtype=complex )
    # Матрица из поднесуших
    for k in range ( Nc ) :
        sk_k=1/np.sqrt (T) * np.exp(1j *2*np.pi *k* df * t )
    sc_matr [ k , : ] = sk_k
    #sd − вектор Nc передаваемых комплексных символов
    sd=np.sign( np.random.rand ( 1 , Nc ) -0.5)+1j *np.sign ( np.random.rand ( 1 , Nc ) -0.5)
    sd=sd.reshape ( Nc )
    xt=np.zeros ( ( 1 , len ( t ) ) , dtype=complex )
    # формирование суммы модулированных поднесущих
    for k in range ( Nc ) :
        sc=sc_matr [k,:]
        xt=xt+sd [k] * sc
    xt=xt.reshape ( Nc )
    # реальная часть сформированного OFDM символа
    # plt.figure ( 2 )
    # plt.title("формирование суммы модулированных поднесущих")
    # plt.plot ( t , xt.real)


    #формирование OFDM символа при помощи ОДПФ
    xt2=np.fft.ifft ( sd , 16 )
    # реальная часть сформированного OFDM символа
    # plt.figure ( 3 )
    # plt.title("формирование OFDM символа при помощи ОДПФ")
    # plt.plot (t,xt2.real)
    n=3
    #прием символа на n поднесущей в виде интеграла от
    # произведения принятого символа на опорное колебание на n
    # поднесущей
    sr=ts *np.sum( xt *np.conjugate(sc_matr[n,:]))
    #прием символа на Nc поднесущих при помощи ДПФ принятого символа
    sr2=np.sqrt(T)/Nc*np.fft.fft(xt)

    # plt.figure(4)
    # plt.title("прием символа на Nc поднесущих при помощи ДПФ принятого символа")
    # plt.plot(sr2)
    # print(sr2)
    arr = np.empty([1,1])
    for i in range(len(sc_matr)):
        arr = np.append(arr, np.sqrt(T)/Nc*np.fft.fft(xt))

    # plt.figure(5)
    # plt.title("прием символа на Nc поднесущих при помощи ДПФ принятого символа")
    # plt.plot(np.abs(arr))

    # plt.show()
    return arr
