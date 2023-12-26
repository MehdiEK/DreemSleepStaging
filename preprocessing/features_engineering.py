"""
File containing classes used to preprocess data.  

Creation date: 22/12/2023
Last modification: 24/12/2023
By: Mehdi EL KANSOULI
"""
import pandas as pd 
import numpy as np 
import scipy.stats as stats
import pywt
import matplotlib.pyplot as plt 

from scipy.fft import fft


class BasicFeaturesCreation():
    """
    Transform time series data into spectral features.
    """

    def __init__(self, sampling_rate, name):
        """ 
        Initialization of the class
        
        :params sampling_rate: int 
            Sampling rate of the recording 
        :params min_freq: int 
            index of the first harmonic to take into account
        :params max_freq: int
            Index of the last harmonic to take into account.        
        """
        self.sampling_rate = sampling_rate
        self.name = "_" + name


    def time_features(self, df):
        """
        Given signals, return time related features

        :params df: pd.DataFrame
        :return pd.DataFrame
        """
        # aggregation functions
        aggregations = {
            "mean": np.mean, 
            "min": np.min, 
            "max": np.max, 
            "std": np.std, 
            "var": np.var, 
            "Rms": lambda X: np.sqrt(np.mean(np.square(X))), 
            "Power": lambda X: np.mean(np.square(X)), 
            "Peak": lambda X: np.max(np.abs(X)), 
            "PTP": lambda X: np.ptp(X), 
            "CrestFactor": lambda X: np.max(np.abs(X))/np.sqrt(np.mean(X**2)), 
            "Skew": lambda X: stats.skew(X),
            "Kurtosis": lambda X: stats.kurtosis(X),  
            "FormFactor": lambda X: np.sqrt(np.mean(X**2))/np.mean(X), 
            "PulseIndiicator": lambda X: np.max(np.abs(X))/np.mean(X)
        }

        output = pd.DataFrame()
        for key, value in aggregations.items():
            output[key + self.name] = df.apply(func=value, axis=1)
        return output



    def frequency_features(self, df):
        """
        Given signals return frequency related features
        """
        # apply fourier transform
        fourier_tf = lambda X: list(np.abs(np.square(fft(list(X)))) / len(X))
        df_fourier = df.apply(func=fourier_tf, axis=1)

        aggregations = {
            "Max_freq": np.max, 
            "Sum_freq": np.sum, 
            "Mean_freq": np.mean, 
            "Var_freq": np.var,
            "Peak_freq": lambda X: np.max(np.abs(X)), 
            "Skew_freq": lambda X: stats.skew(X), 
            "Kurtosis_freq": lambda X: stats.kurtosis(X)
        }

        output = pd.DataFrame()
        for key, value in aggregations.items():
            output[key + self.name] = df_fourier.apply(func=value)
        return output


    def transform(self, X):
        """
        Takes as input a time series and uses fast fourier 
        transform to compute new features.

        :params X: pd.DataFrame
            Columns correspond to recording at different time t and index are
            the different data points.
        
        :return pd.DataFrame
            Columns are new "spectral" features.
        """
        # aggregates time related features
        df_time = self.time_features(X)

        # aggregates frequency related features
        df_freq = self.frequency_features(X)

        return df_freq.join(df_time)
    

class WaveletTransformer():
    """
    Perform a wavelet transform of signals 
    """
    def __init__(self, wavelet='cmor', scales=range(1, 128), sampling_rate=1):
        """
        Initialization of the class
        """
        self.wavelet = wavelet
        self.scales = scales
        self.sampling_rate = sampling_rate

    def wavelet_features(self, signal):
        """
        Wavelet decomposition of the signal. 
        """
        coefficients, frequencies = pywt.cwt(signal, self.scales, self.wavelet, sampling_period=1/self.sampling_rate)
        return coefficients, frequencies
    
    def transform(self, X):
        """
        Takes as input a time series and uses fast fourier 
        transform to compute new features.

        :params X: pd.DataFrame
            Columns correspond to recording at different time t and index are
            the different data points.
        
        :return pd.DataFrame
            Columns are new "spectral" features.
        """
        # aggregates time related features
        df_time = self.time_features(X)

        # aggregates frequency related features
        df_freq = self.frequency_features(X)

        return df_freq.join(df_time)

    def plot_cwt(self, signal, time, coefficients, frequencies, title='Continuous Wavelet Transform'):
        plt.figure(figsize=(12, 8))
        plt.imshow(np.abs(coefficients), extent=[time.min(), time.max(), frequencies[-1], frequencies[0]], aspect='auto', cmap='jet')
        plt.title(title)
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        plt.colorbar(label='Magnitude')
        plt.show()


class ShortTimeFourierTransformer():
    pass


if __name__ == '__main__':
    # define file path 
    file_path = "./../data/training_records/dreem_0.npy"

    # load data
    data = np.load(file_path)
    df = pd.DataFrame(data)
    df.set_index(0, inplace=True)

    category = {
        "First_EEG": (1, 7501, 250), 
        "Second_EEG": (7501, 15001, 250), 
        "Third_EEG": (15001, 22501, 250), 
        "Fourth_EEG": (22501, 30001, 250), 
        "Fifth_EEG": (30001, 37501, 250), 
        "X_axis": (37501, 39001, 50), 
        "Y_axis": (39001, 40501, 50), 
        "Z_axis": (40501, 42001, 50) 
    }

    cat = "X_axis" 
    min_, max_, sampling_rate = category.get(cat)
    tf_1stEEG = BasicFeaturesCreation(sampling_rate=sampling_rate, 
                                      name=cat)
    # df_transformed = tf_1stEEG.transform(df.iloc[:, min_-1:max_-1])
    df_transformed = tf_1stEEG.transform(df[np.arange(min_, max_)])

    print(df.info())
    print(df_transformed.info())
    print(df_transformed.head())