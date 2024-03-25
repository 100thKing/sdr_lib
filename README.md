# Указание по работе с библиотекой

##### Содержание
- [TODO](#todo)
- [generate_binary_arr](#generate_binary_arr)
- [bin_to_complex](#bin_to_complex)
- [modulate_qpsk](#modulate_qpsk)
- [repeat_elements](#repeat_elements)
- [Add_Barker](#Add_Barker)
- [higher_signal](#higher_signal)
- [signal_to_transmite](#signal_to_transmite)
- [configure_sdr_tx](#configure_sdr_tx)
- [configure_sdr_rx](#configure_sdr_rx)

<a name=“todo”><h2>TODO</h2></a>

- Проверить правильность функций настройки sdr на tx\rx
- Создать функции для приемника sdr rx
- Изменить чтобы на выход шел np.array следующие функции: generate_binary_arr, string_to_bin 
- Определить отличия функций bin_to_complex и modulate_qpsk


<a name=“generate_binary_arr”><h2>generate_binary_arr</h2></a>
- Функция возвращает бинарный массив длинной len, заполненный 0 и 1
- Передаваемые аргументы: int len - число, длина выходного массива

<a name=“bin_to_complex”><h2>bin_to_complex</h2></a>
- Функция возвращает numpy массив комплексных значений (1+1j)
- Передаваемые аргументы: np.array binary - бинарный массив, обязательно четной длины 

<a name=“modulate_qpsk”><h2>modulate_qpsk</h2></a>
- Функция возвращает numpy массив комплексных значений (0,707+0,707j)
- Передаваемые аргументы: np.array bit_array - бинарный массив
 
<a name=“add_pylots”><h2>add_pylots</h2></a>
- Функция вставляет пилоты val в сигнал через каждые pos символов в массив arr
- Передаваемые аргументы: np.array arr - массив с данными
