#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import wave
import os
import struct
import matplotlib.pyplot as plt

def np_to_int16_bytes(x):
    x = np.int16(x * 2**(16-1))
    return struct.pack(str(len(x))+'h', *x)


def int16_bytes_to_np(x, num, sw):
    x = np.array(struct.unpack(str(num)+'h', x))
    x = x.astype(np.float) / 2**(16-1)
    return x


class Signal(object):

    def __init__(self, Fs=44100, duration=None, data=[]):
        self._duration = duration
        self._Fs, self._Ts = Fs, 1./Fs
        self._data = data
        if duration:
            self._n = np.linspace(0, duration, self._Fs*self._duration)

    def estimate_amplitude(self, base):
        amp = (np.dot(self._data, base._data) /
               np.dot(base._data, base._data) )
        return amp


    '''HELPER FUNCTIONS'''

    def dft(self, win_size = 44100, window_function = np.hanning):
        samples = self.window_samples(win_size = win_size)
        if window_function is not None:
            windows = []
            for window in samples:
                func = window_function(len(window))
                windows.append([s*h for s,h in zip(window, func)])
            samples = windows
        output = []
        for window in samples:
            t = np.fft.fft(window, n = win_size)/win_size
            t = t[:len(t)/2]
            output.append(t)
        return output

    def magnitude(self, win_size = 44100):
        data = self.dft(win_size = win_size)
        return [np.abs(d) for d in data]

    def freq_domain(self, win_size = 44100):
        freqs = np.linspace(0, self._Fs, win_size)
        return freqs[:len(freqs)/2]

    def time_domain(self, win_size = 44100):
        num_points = int(self._duration*self._Fs/win_size)
        return np.linspace(0, self._duration, num_points)

    def window_samples(self, win_size = 44100):
        signal = self._data
        return [signal[x:x+win_size] for x in range(0, len(signal)-win_size+1, win_size)]


    '''COMPUTATION FUNCTIONS'''

    def auto_corr(self, win_size = 44100):
        samples = self.window_samples(win_size = win_size)
        freq = []
        for window in samples:
            corr = np.correlate(window, window, mode='full')
            corr = corr[len(corr)//2:]
            d = np.diff(corr)
            start = np.where(d > 0)[0][0]
            peak = np.argmax(corr[start:]) + start
            freq.append(self._Fs/peak)
        return freq

    def centroid_sonification(self, win_size = 44100, s_factor = 5):
        m = self.magnitude(win_size = win_size)
        f = self.freq_domain(win_size = win_size)
        def sonify(data):
            num = 0
            for d, freq in zip(data,f):
                num += d*freq
            denum = np.sum(data)
            return num/denum
        centroids = [sonify(window) for window in m]
        sums = []
        if s_factor > 0:
            vals = []
            avg_centroids = []
            for c in range(len(centroids)):
                if len(vals) >= s_factor:
                    vals.pop(0)
                vals.append(centroids[c])
                avg_centroids.append(np.average(vals))
            centroids = avg_centroids
        return centroids

    def fund_freq(self, win_size = 44100):
        m = self.magnitude(win_size = win_size)
        f = self.freq_domain(win_size = win_size)
        t = self.time_domain(win_size = win_size)
        sample_max = []
        for sample in m:
            sample_max.append(f[sample.argmax(axis = 0)])
        return sample_max


    '''PLOT FUNCTIONS'''

    def plot(self):
        if len(self._data):
            plt.plot(self._n, self._data)
            plt.show()
            plt.close()

    def plot_data(self, x, y, title = '', xlabel = '', ylabel = ''):
        fig, ax = plt.subplots(1, 1)
    	ax.plot(x, y)
    	ax.set_title(title)
    	ax.set_xlabel(xlabel)
    	ax.set_ylabel(ylabel)
        plt.show()
        plt.close()

    def plot_auto_corr(self, win_size = 44100):
        t = self.time_domain(win_size = win_size)
        freq = self.auto_corr(win_size = win_size)
        self.plot_data(t, freq, xlabel = 'Time (s)', ylabel = 'Frequency (Hz)')

    def plot_centroid_sonification(self, win_size = 44100, s_factor = 5):
        t = self.time_domain(win_size = win_size)
        centroid = self.centroid_sonification(win_size = win_size, s_factor = s_factor)
        self.plot_data(t, centroid, xlabel = 'Time (s)', ylabel = 'Centroid')

    def plot_fund_freq(self, win_size = 44100):
        t = self.time_domain(win_size = win_size)
        sample_max = self.fund_freq(win_size = win_size)
        self.plot_data(t, sample_max, xlabel = 'Time (s)', ylabel = 'Fundamental Frequency (Hz)')

    def plot_magnitude(self, window = 0, win_size = 44100):
        m = self.magnitude(win_size = win_size)
        f = self.freq_domain(win_size = win_size)
        assert(window < len(m))
        self.plot_data(f, m, xlabel = 'Frequency  (Hz)', ylabel = 'Magnitude')

    def plot_sum(self, win_size = 44100):
        corr = self.auto_corr(win_size = win_size)
        freq = self.fund_freq(win_size = win_size)
        sum = [(c+f)/2 for c,f in zip(corr,freq)]
        t = self.time_domain(win_size = win_size)
        self.plot_data(t, sum, xlabel = 'Time (s)', ylabel = 'Sum')


    '''SIGNAL I/O FUNCTIONS'''

    def centroid_to_signal(self, win_size = 44100, s_factor = 5):
        centroids = self.centroid_sonification(win_size = win_size, s_factor = s_factor)
        signals = []
        for centroid in centroids:
            signals.append(Sinusoid(freq = centroid, duration = win_size*1./self._Fs))
        return Sequence(*signals)

    def wav_write(self, outfile, Nch=1, Sw=2, normalize=True):
        if len(self._data):
            x = self._data
            x = x / max(x) if normalize else x
            dst = wave.open(outfile, 'wb')
            dst.setparams((Nch, Sw, self._Fs, len(x), 'NONE', 'not_compressed'))
            dst.writeframes(np_to_int16_bytes(x))
            dst.close()

    def wav_read(self, in_file):
        assert(os.path.exists(in_file))
        src = wave.open(in_file, 'rb')
        nch, sw, fs, nframes, _, _ = src.getparams()
        duration = nframes/fs
        nframes = fs*duration + 1
        self.__init__(Fs=fs, duration=duration)
        assert(nch == 1), "wav must be 1 ch"
        self._data = int16_bytes_to_np(src.readframes(nframes), nframes, sw)
        src.close()


    '''SIGNAL MODIFICATION FUNCTIONS'''

    def add_signal(self, signal, start_time = 0):
        assert(signal.params[1] == self._Fs)
        if start_time + signal._duration > self._duration:
            self._data = [self.data[i] if i < len(self.data) else 0
                          for i in range(int((start_time + signal._duration)*self._Fs))]
            self._duration = signal._duration + start_time
            self._n = np.arange(0, self._duration, self._Ts)
        start, stop = int(start_time*self._Fs), int((start_time + signal._duration)*self._Fs)
        for i in range(start, stop):
            self._data[i] += signal._data[i - start]

    def multiply_signal(self, signal):
        assert(len(signal) == len(self._data))
        self._data = [self.data[i]*signal[i] for i in range(len(self._data))]


    @property
    def data(self):
        return self._data

    @property
    def duration(self):
        return self._duration

    @property
    def params(self):
        return self._duration, self._Fs, self._Ts


class Sinusoid(Signal):

    def __init__(self, duration=1, Fs=44100.0, amp=1.0, freq=440.0, phase=0):
        super(self.__class__, self).__init__(duration=duration, Fs=Fs)
        self.A, self.f, self.phi = amp, freq, phase
        self._w = 2 * np.pi * self.f
        self.__make()

    def __make(self):
        self._data = self.A * np.sin(self._w * self._n + self.phi)

    def power(self):
        return self.A**2/2.0

    def add_noise(self, snr):
        sigma2 = self.power()/(10**(snr/10.0))
        noise = np.random.normal(0, np.sqrt(sigma2), len(self._data))
        self._data += noise

    def remove_noise(self):
        self.__make()

    def shift(self, phi):
        self._phi = phi
        self.__make()

class Sequence(Signal):

    def __init__(self, *signals):
        _, fs, _ = signals[0].params
        duration = np.sum([sig.duration for sig in signals])
        super(self.__class__, self).__init__(duration=duration, Fs=fs)
        self._data = np.hstack([sig.data for sig in signals])


class MidiNotes():
    TUNING = 440
    NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self):
        self._notes = {}
        for i in range(24, 109):
            f = 2**((i-69)/12.0) * self.TUNING
            self._notes[self.NAMES[i % 12] + str(i/12-1)] = f

    def freq(self, name):
        return self._notes[name]
