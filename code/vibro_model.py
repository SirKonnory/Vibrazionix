import numpy as np
from scipy import signal, fft
from scipy.linalg import toeplitz, eig
from config import Config

config = Config()


class ModelClass:

    def __init__(self):

        self.recovered = None
        self.fft = None
        self.data_75_percent = None
        self.data_50_percent = None
        self.data_25_percent = None
        self.data_var = None
        self.data_mean = None
        self.data_rms = None
        self.data_std = None
        self.data_max = None
        self.data_min = None
        self.actual_col = None
        self.t = None
        self.pc = None
        self.ff = None
        self.tt = None
        self.zz = None
        self.psd = None
        self.f = None
        self.observers = []
        self.data = None
        self.fs = None
        self.origin_data = None

    def load_data(self, data, fs):

        self.data = data
        self.origin_data = np.copy(data)

        self.fs = fs
        dt = 1 / self.fs

        self.t = np.arange(0, self.data.shape[0] * dt, dt)
        self.actual_col = [i for i in range(self.data.shape[1])]

    def delete_data(self, col):
        self.actual_col.remove(col)
        self.data = self.origin_data[:, self.actual_col]

    def insert_data(self, col):
        self.actual_col.append(col)
        self.data = self.origin_data[:, self.actual_col]

    def calc_integrate_param(self):

        self.data_min = np.around(self.data.min(
            axis=0), decimals=config.round_decimal)
        self.data_max = np.around(self.data.max(
            axis=0), decimals=config.round_decimal)
        self.data_std = np.around(
            np.std(self.data, axis=0), decimals=config.round_decimal)
        self.data_rms = np.around(
            np.sqrt(np.mean(self.data ** 2, axis=0)), decimals=config.round_decimal)
        self.data_mean = np.around(
            np.mean(self.data, axis=0), decimals=config.round_decimal)
        self.data_var = np.around(
            np.var(self.data, axis=0), decimals=config.round_decimal)
        self.data_25_percent = np.around(np.percentile(
            self.data, 25, axis=0), decimals=config.round_decimal)
        self.data_50_percent = np.around(np.percentile(
            self.data, 59, axis=0), decimals=config.round_decimal)
        self.data_75_percent = np.around(np.percentile(
            self.data, 75, axis=0), decimals=config.round_decimal)

    def get_fft(self):
        if self.actual_col:
            N = self.data.shape[0]

            self.fft = np.abs(fft.fft2(self.data, axes=[0]) / N)[0:N // 2]

            self.f = fft.fftfreq(N, 1 / self.fs)[:N // 2]

    def get_stft(self, data, fs, window_type, window_size, overlap):

        if self.actual_col:

            self.ff, self.tt, self.zz = signal.stft(data, fs, window_type, window_size, overlap)
            self.zz = np.abs(self.zz)

    def get_psd(self, fs, window_type, window_width, overlap):

        self.psd = []

        for i in range(self.data.shape[1]):

            self.f, temp = signal.welch(self.data[:, i], fs, window_type, window_width, overlap)
            self.psd.append(temp)

        self.psd = np.array(self.psd).transpose()

    def get_spectogram(self, data, fs, window_type, window_size, overlap):

        self.ff, self.tt, self.spectorgam = signal.spectrogram(self.data[0], fs=fs,
                                                               window=window_type, nperseg=window_size,
                                                               noverlap=int(window_size * overlap / 100))

        for i in range(1, self.data.shape[0]):
            _, _, spectorgam_temp = signal.spectrogram(self.data[i], fs,
                                                       window=window_type, nperseg=window_size,
                                                       noverlap=int(window_size * overlap / 100))
            self.spectorgam = np.vstack([self.spectorgam, spectorgam_temp])

        # Sxx, f_a, t_a, fig = pyspecgram.pyqtspecgram(self.data, window_size, fs, Fc=0)

    def filt_signal_data(self, lowcut_f, topcut_f, rank, df):

        if lowcut_f.isdigit() and (not topcut_f.isdigit()):

            lowcut_f = int(lowcut_f) / (df / 2)
            sos = signal.butter(rank, lowcut_f, btype='lowpass', output='sos')

            self.data = signal.sosfilt(sos, self.data, axis=0)

        elif topcut_f.isdigit() and (not lowcut_f.isdigit()):

            topcut_f = int(topcut_f) / (df / 2)
            sos = signal.butter(
                rank, topcut_f, btype='highpass', output='sos')

            self.data = signal.sosfilt(sos, self.data, axis=0)

        elif topcut_f.isdigit() and lowcut_f.isdigit():

            lowcut_f = int(lowcut_f) / (df / 2)
            topcut_f = int(topcut_f) / (df / 2)
            sos = signal.butter(
                rank, [lowcut_f, topcut_f], btype='bandstop', output='sos')

            self.data = signal.sosfilt(sos, self.data, axis=0)

    def reduce_signal_data(self, k):

        if k.isdigit():
            k = int(k)
            self.t = self.t[::k]
            self.data = self.data[:, ::k]
            # self.notify_observers()

    def quantization_signal_data(self, levels):

        if levels.isdigit():
            levels = int(levels)
            signal_range = np.max(self.data) - np.min(self.data)
            step = signal_range / (levels - 1)

            self.data = np.round(self.data / step) * step
            # self.notify_observers()

    def smooth_signal_data(self, window_size):

        if window_size.isdigit():

            window_size = int(window_size)
            window = np.ones(window_size) / window_size
            temp = np.array([np.convolve(self.data[0], window, mode='same')])

            for i in range(1, self.data.shape[0]):
                temp = np.vstack([temp, [np.convolve(self.data[i], window, mode='same')]])

            self.data = temp
            self.notify_observers()

    def pca_compute(self, data, M, start, end, method):

        N = self.data.shape[0]
        X = data

        # X = X - np.mean(X)
        # X = X / np.std(X, ddof=1)

        if True:  # method == 0
            # Метод Гусеницы
            Y = np.zeros((N - M + 1, M))

            for m in range(M):
                Y[:, m] = X[m: N - M + m + 1].reshape(-1)

            self.cov = np.dot(Y.T, Y) / (N - M + 1)

        # else:
        #     # Метод матрицы Тёплица
        #     covX = np.correlate(X, X, mode='full') / len(X)
        #     covX = covX[len(X) - M:len(X)]
        #
        #     self.cov = toeplitz(covX)

        # Вычисление собственных векторов и собственных значений
        self.lamb, self.rho = eig(self.cov)

        # Вычисление главных компонент
        self.pc = Y @ self.rho

        # Восстановление компонент
        self.rc = np.zeros((N, M))

        for m in range(M):

            buf = np.outer(self.pc[:, m], self.rho[:, m])
            buf = np.fliplr(buf)
            for n in range(N):  # Анти-диагональное усреднение
                self.rc[n, m] = np.mean(np.diag(buf, - (N - M) + n))

        self.rc = np.flip(self.rc)

        if start.isdigit() * end.isdigit():
            self.recovered = np.sum(self.rc[:, start: end], axis=1)
        else:
            self.recovered = np.sum(self.rc[:, :], axis=1)

    def generate_data(self, a: list, f: list, p: list, noise: float, t: float, fs: float, n: int):
        self.data = []

        self.fs = fs
        dt = 1 / self.fs

        self.t = np.arange(0, t, dt)

        np.sin(60 * np.pi / 180 + 90 * np.pi / 180)

        for i in range(n):
            sum_sin = 0
            for j in range(3):
                sin = a[j] * np.sin(2 * np.pi * (f[j] * self.t + p[j] / 2 / np.pi))
                sum_sin += sin
            noise_sum = noise * np.random.randn(int(t / dt))
            self.data.append(sum_sin + noise_sum)

        self.data = np.transpose(np.array(self.data))

        self.origin_data = np.copy(self.data)

        self.actual_col = [i for i in range(self.data.shape[1])]
