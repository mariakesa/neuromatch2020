import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler

def one_receptive_field(data_path,time_series):

    dt=1
    iframe = np.load(data_path+'iframe.npy') # iframe[n] is the microscope frame for the image frame n
    ivalid = iframe+dt<time_series.shape[-1] # remove timepoints outside the valid time range
    iframe = iframe[ivalid]
    mov=np.load(data_path+'mov.npy')
    mov = mov[:, :, ivalid]
    ly, lx, nstim = mov.shape

    mov=mov[:,:,:30000]

    NT=30000

    time_series=time_series[:30000]

    X = np.reshape(mov, [-1, NT]) # reshape to Npixels by Ntimepoints
    X = X-0.5 # subtract the background
    X = np.abs(X) # does not matter if a pixel is black (0) or white (1)
    X = zscore(X, axis=1)/NT**.5  # z-score each pixel separately
    npix = X.shape[0]

    lam = 0.1
    #ncomps = Sp.shape[0]
    B0 = np.linalg.solve((X@ X.T + lam * np.eye(npix)),  (X @ time_series)) # get the receptive fields for each neuron

    B0 = np.reshape(B0, (ly, lx, 1))
    B0 = gaussian_filter(B0, [.5, .5, 0])

    return B0

def show_receptive_field(B0,tau):
    rf = B0[:,:,0]
    rfmax = np.max(B0)
    # rfmax = np.max(np.abs(rf))
    plt.title('Receptive field with '+str(tau)+' timeconstant')
    plt.imshow(rf, aspect='auto', cmap = 'bwr', vmin = -rfmax, vmax = rfmax)
    plt.show()
