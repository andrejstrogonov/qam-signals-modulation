import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile

plt.rc('font', family='Sawasdee', weight='bold')
plt.rc('font', family='Garuda')
plt.rc('axes', unicode_minus=False)


###########################################
class Signal(object):

    #######################################
    def __init__(self, duration=1.0, sampling_rate=22050, func=None):
        """Инициализировать объект сигнала с указанной продолжительностью (в секундах)
        и частоту дискретизации (в Гц). Если функция предоставлена, сигнал/
        данные будут инициализированы значениями этой функции для всей
        продолжительности"""
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.freqs = np.arange(int(duration * sampling_rate), dtype=complex)
        self.freqs[:] = 0j
        if func is not None:
            self.sample_time_function(func)

    #######################################
    def read_wav(self, wav_file, channel='left'):
        """Считать данные из указанного wav-фвйла в сигнальный объект. Для
        стереопотока, может быть извлечен только один канал («левый» или «правый»)."""

        rate, data = wavfile.read(wav_file)
        n = data.shape[0]
        self.sampling_rate = rate
        self.duration = float(n) / rate

        if data.dtype == np.dtype('int16'):
            normalizer = 32768.0
        elif data.dtype == np.dtype('int8'):
            normalizer = 256.0
        else:
            raise (Exception('Unsupport data type'))

        if len(data.shape) == 2:  # stereo stream
            if channel == 'left':
                data = data[:, 0]
            elif channel == 'right':
                data = data[:, 1]
            else:
                raise (Exception('Invalid channel choice "%s"' % channel))

        self.freqs = fft(data / normalizer)

    #######################################
    def write_wav(self, wav_file):
        """Запишите данные сигнала в указанный волновой файл, используя тип данных int16"""
        wavfile.write(
            wav_file,
            self.sampling_rate,
            (ifft(self.freqs).real * 32768).astype(np.dtype('int16')))

    #######################################
    def get_sampling_rate(self):
        """Возвращает частоту дискретизации, связанную с сигналом в Гц."""
        return self.sampling_rate

    #######################################
    def get_duration(self):
        """Возвращает продолжительность сигнала в секундах"""
        return self.duration

    #######################################
    def amplify(self, factor):
        """Задание коэффициента усилиения"""
        self.freqs *= factor

    #######################################
    def clear(self, cond=lambda f: True):
        """Установите амплитуды всех частот, удовлетворяющих условию, cond, равным
        ноль, где cond — логическая функция, принимающая частоту в Гц."""
        n = len(self.freqs)
        for i in range(n):
            # преобразование индекса в соответствующее значение частоты
            f = float(i) * self.sampling_rate / n
            if cond(f):
                self.freqs[i] = 0j

    #######################################
    def set_freq(self, freq, amplitude, phase=0):
        """Установите конкретную частотную составляющую с заданной амплитудой и
        фазовый сдвиг (в градусах) к сигналу"""
        n = len(self.freqs)

        # вычислить индекс, по которому указанная частота находится в массиве
        index = int(np.round(float(freq) * n / self.sampling_rate))

        # распределить амплитуду сигнала по действительной и мнимой осям
        re = float(n) * amplitude * np.cos(phase * np.pi / 180.0)
        im = float(n) * amplitude * np.sin(phase * np.pi / 180.0)

        # равномерно распределить компонент переменного тока по положительным и отрицательным частотам
        if freq != 0:
            re = re / 2.0
            im = im / 2.0

            # чтобы обеспечить сигнал во временной области с действительным знаком
            #  две части должны быть комплексно сопряжены друг с другом.
            self.freqs[index] = re + 1j * im
            self.freqs[-index] = re - 1j * im

        else:
            # Компонент постоянного тока имеет только одну часть
            self.freqs[index] = re + 1j * im

    #######################################
    def sample_time_function(self, func):
        """
        Выборочные значения из функции с действительной частью во временной области, func(t), где
        t будет указано в секундах. Экземпляры собираются по
        частоте дискретизации, связанные с объектом Signal.
        """
        n = len(self.freqs)
        signal = np.arange(n, dtype=float)
        for i in range(n):
            signal[i] = func(float(i) / self.sampling_rate)
        self.freqs = fft(signal)

    ###########################################
    def square_wave(self, freq, flimit=8000):
        """
        Генерация прямоугольного сигнала с ограниченным диапазоном на сигнальном объекте
        """
        self.clear()
        f = freq
        while f <= flimit:
            self.set_freq(f, 1.0 / f, -90)
            f += 2 * freq

    #######################################
    def get_time_domain(self):
        """
        Возвращает кортеж (X, Y), где X — массив, хранящий ось времени,
        и Y представляет собой массив, хранящий представление сигнала во временной области.
        """
        x_axis = np.linspace(0, self.duration, len(self.freqs))
        y_axis = ifft(self.freqs).real
        return x_axis, y_axis

    #######################################
    def get_freq_domain(self):
        """
        Возвращает кортеж (X,A,P), где X — массив, хранящий ось частот.
        до частоты Найквиста (исключая отрицательную частоту), а А и
        P — массивы, хранящие амплитуду и фазовый сдвиг (в градусах) каждого
        частота
        """
        n = len(self.freqs)
        num_freqs = int(np.ceil((n + 1) / 2.0))
        x_axis = np.linspace(0, self.sampling_rate / 2.0, num_freqs)

        # извлекать только положительные частоты и масштабировать их так,
        # чтобы величина не зависела от длины массива
        a_axis = abs(self.freqs[:num_freqs]) / float(n)
        p_axis = np.arctan2(
            self.freqs[:num_freqs].imag,
            self.freqs[:num_freqs].real) * 180.0 / np.pi

        # двойные амплитуды составляющих переменного тока
        # (поскольку мы отбросили отрицательные частоты)
        a_axis[1:] = a_axis[1:] * 2

        return x_axis, a_axis, p_axis

    #######################################
    def shift_freq(self, offset):
        """
        Сдвиг сигнала в частотной области на величину, заданную смещением
        (в Гц). Если смещение положительное, сигнал смещается вправо
        по оси частот. Если смещение отрицательное, сигнал
        смещается влево по оси частот.
        """
        n = len(self.freqs)
        nyquist = n / 2

        # вычислить индекс на основе массива из указанного смещения в Гц
        offset = int(np.round(float(offset) * n / self.sampling_rate))
        if abs(offset) > nyquist:
            raise Exception(
                'Shifting offset cannot be greater than the Nyquist frequency')

        if offset > 0:
            self.freqs[offset:nyquist] = np.array(self.freqs[:nyquist - offset])
            self.freqs[:offset] = 0

            self.freqs[-nyquist + 1:-offset] = np.array(self.freqs[-(nyquist - offset) + 1:])
            self.freqs[-offset + 1:] = 0
        else:
            offset = -offset
            self.freqs[:nyquist - offset] = np.array(self.freqs[offset:nyquist])
            self.freqs[nyquist - offset:nyquist] = 0

            self.freqs[-(nyquist - offset) + 1:] = np.array(self.freqs[-nyquist + 1:-offset])
            self.freqs[-nyquist + 1:-nyquist + offset] = 0

    #######################################
    def shift_time(self, offset):
        """
        Сдвиг сигнала во временной области на величину, заданную смещением
        (в секундах). Если смещение положительное, сигнал смещается в
        прямо по оси времени. Если смещение отрицательное, сигнал
        смещены влево по оси времени.
        """
        noff = offset * self.sampling_rate
        x, y = self.get_time_domain()
        if noff > 0:
            y[noff:] = y[:len(x) - noff].copy()
            y[:noff] = 0.0
        elif noff < 0:
            noff = -noff
            y[:len(x) - noff] = y[noff:].copy()
            y[len(x) - noff:] = 0.0
        self.freqs = fft(y)

    #######################################
    def copy(self):
        """
        Клонирование сигнальный объект в другой идентичный сигнальный объект.
        """
        s = Signal()
        s.duration = self.duration
        s.sampling_rate = self.sampling_rate
        s.freqs = np.array(self.freqs)
        return s

    #######################################
    def mix(self, signal):
        """
        Смешайте сигнал с другим данным сигналом. Частота дискретизации и продолжительность
        обоих сигналов должны совпадать.
        """
        if self.sampling_rate != signal.sampling_rate \
                or len(self.freqs) != len(signal.freqs):
            raise Exception(
                'Signal to mix must have identical sampling rate and duration')

        self.freqs += signal.freqs

    #######################################
    def __add__(self, s):
        newSignal = self.copy()
        newSignal.mix(s)
        return newSignal

    #######################################
    def plot(self, dB=False, phase=False, stem=False, frange=(0, 10000)):
        """
        Создайте три подграфика, показывающая распределение амплитуды по времени (как амплитуду, так и
        фаза) и представления данного сигнала во временной области.
        Если разделитель имеет значение True, стержневые графики будут использоваться как для амплитуды, так и для фазы.
        Если dB равно True, будет показана амплитуда на графике в частотной области с логарифмической шкалой.
        Если для фазы установлено значение True, также будет создан график фазового сдвига.
        """
        plt.subplots_adjust(hspace=.4)

        if phase:
            num_plots = 3
        else:
            num_plots = 2

        # постройте сигнал временной области
        plt.subplot(num_plots, 1, 1)
        plt.cla()
        x, y = self.get_time_domain()
        plt.grid(True)
        plt.xlabel(u'Time (s)')
        plt.ylabel('Value')
        plt.plot(x, y, 'g')

        # АЧХ
        x, a, p = self.get_freq_domain()
        start_index = int(float(frange[0]) / self.sampling_rate * len(self.freqs))
        stop_index = int(float(frange[1]) / self.sampling_rate * len(self.freqs))
        x = x[start_index:stop_index]
        a = a[start_index:stop_index]
        p = p[start_index:stop_index]
        plt.subplot(num_plots, 1, 2)
        plt.cla()
        plt.grid(True)
        plt.xlabel(u'Frequency (Hz)')

        if dB:
            a = 10. * np.log10(a + 1e-10) + 100
            plt.ylabel(u'Amplitude (dB)')
        else:
            plt.ylabel(u'Amplitude')

        if stem:
            plt.stem(x, a, 'b')
        else:
            plt.plot(x, a, 'b')

        # ФЧХ
        if phase:
            plt.subplot(num_plots, 1, 3)
            plt.cla()
            plt.grid(True)
            plt.xlabel(u'Frequency (Hz)')
            plt.ylabel(u'Phase (degree)')
            plt.ylim(-180, 180)
            if stem:
                plt.stem(x[start_index:stop_index], p[start_index:stop_index], 'r')
            else:
                plt.plot(x[start_index:stop_index], p[start_index:stop_index], 'r')

        plt.show()


###########################################
def test1():
    """
    генерация прямоугольной волны 5 Гц с частотой среза 50 Гц
    затем демонстрация сигнала временной области
    """
    s = Signal()
    s.square_wave(5, flimit=50)
    x, y = s.get_time_domain()
    plt.plot(x, y)
    plt.grid(True)
    plt.show()


###########################################
def test2():
    """
    генерация прямоугольную волну 2 Гц с частотой среза 50 Гц
    затем отобразите сигнал как во временной, так и в частотной области
    """
    s = Signal()
    s.square_wave(2, flimit=50)
    s.plot(stem=True, phase=True, frange=(0, 50))


###########################################
def test3():
    """
    генерация составной сигнал, содержащий синусоидальные волны 3 Гц и 2 Гц
    """

    def test_func(t):
        return 0.2 * np.sin(2 * np.pi * t * 3) + 0.3 * np.sin(2 * np.pi * t * 2)

    s = Signal(func=test_func)
    s.plot(frange=(0, 10), stem=True)


###########################################
def test4():
    """
    генерация DTMF (двухтональный многочастотный) сигнал, представляющий клавиатуру '2'
    затем запись выходной волны в файл
    """
    s = Signal()
    s.set_freq(770, .3, 0)
    s.set_freq(1336, .3, 0)
    s.plot(frange=(0, 1500), stem=False)
    s.write_wav('2.wav')


###########################################
def test5():
    """
    Считывание волнового файла, содержащего форму волны DTMF клавиатуры '6', и отображение его
    сигнал и частотный спектр
    """
    s = Signal()
    s.read_wav('Dtmf6.wav')
    s.plot(frange=(0, 2000), stem=False)


###########################################
def test6():
    """
    Тестовый сдвиг частоты и микширование сигналов
    """
    s1 = Signal()
    s1.set_freq(50, .3)
    s2 = s1.copy()
    #s2.shift_freq(-30)
    s1.mix(s2)
    #s2.shift_freq(70)
    s1.mix(s2)
    s1.plot(stem=True, phase=False, frange=(0, 100))


if __name__ == '__main__':
    test6()
