from __future__ import division
from __future__ import absolute_import
import cv2
import numpy as np
import time
from scipy import signal

'''
    La classe signal_processing contiente tutte le funzioni necessarie per la manipolazione del segnale luminoso.
'''


class Signal_processing(object):
    def __init__(self):
        self.a = 1

    def extract_color(self, ROIs):
        """
        :param ROIs: sono le ROI identificate da ROI_extraction()
        :return: output_val: media del colore nelle ROI

            Estrazione del colore dalle ROI
        """

        g = []
        for ROI in ROIs:
            g.append(np.mean(ROI[:, :, 1]))

        output_val = np.mean(g)
        return output_val

    def normalization(self, data_buffer):
        u'''
        normalize the input data buffer
        '''

        # normalized_data = (data_buffer - np.mean(data_buffer))/np.std(data_buffer)
        normalized_data = data_buffer / np.linalg.norm(data_buffer)

        return normalized_data

    def signal_detrending(self, data_buffer):

        detrended_data = signal.detrend(data_buffer)

        return detrended_data

    def interpolation(self, data_buffer, times):

        L = len(data_buffer)

        even_times = np.linspace(times[0], times[-1], L)

        interp = np.interp(even_times, times, data_buffer)
        interpolated_data = np.hamming(L) * interp
        return interpolated_data

    def fft(self, data_buffer, fps):
        """
            Trasformata di Fourier

        :param data_buffer: array con i valori di tutti i battiti cardiaci in un preciso momento
        :param fps: fps del video
        """

        L = len(data_buffer)

        freqs = float(fps) / L * np.arange(L / 2 + 1)

        freqs_in_minute = 60. * freqs

        raw_fft = np.fft.rfft(data_buffer * 30)
        fft = np.abs(raw_fft) ** 2

        interest_idx = np.where((freqs_in_minute > 65) & (freqs_in_minute < 110))[0]

        interest_idx_sub = interest_idx[:-1].copy()  #
        freqs_of_interest = freqs_in_minute[interest_idx_sub]

        fft_of_interest = fft[interest_idx_sub]

        return fft_of_interest, freqs_of_interest

    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):
        u'''

        '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype=u'band')

        filtered_data = signal.lfilter(b, a, data_buffer)

        return filtered_data
