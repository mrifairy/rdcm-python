#
# Following closely the code of TAPAS rDCM
#

import numpy as np
from .get_convolution_bm import get_convolution_bm
from .euler_make_indices import euler_make_indices
from .euler_gen import euler_gen


def generate(DCM, options, SNR):
  '''
  Generates synthetic fMRI data under a given signal to noise ratio (SNR) 
  with the fixed hemodynamic convolution kernel

    Input:
      DCM         - model structure
      options     - estimation options
      SNR         - signal to noise ratio
    Output:
      DCM         - model structure with generated synthetic time series
  '''

  # compile source code of integrator
  # tapas_rdcm_compile()

  # Setting parameters
  if options is not None and options.y_dt:
      DCM.Y.dt = options.y_dt

  r_dt = DCM.Y.dt/DCM.U.dt
  N = DCM.U.u.shape[0] # DCM.U.u is sparse?
  nr = DCM.a.shape[0]

  # specify the array for the data
  y = np.zeros(( N, nr ))

  # generate fixed hemodynamic response function (HRF)
  if not hasattr(options,'h') or len(options.h) != DCM.U.u.shape[0]:
      options.DCM         = DCM
      options.conv_dt     = DCM.U.dt
      options.conv_length = DCM.U.u.shape[0]
      options.conv_full   = 'true'
      options.h           = get_convolution_bm(options)


  # get the hemodynamic response function (HRF)
  h = options.h

  # Getting neuronal signal (x)
  DCM.U.u = np.vstack([ DCM.U.u, DCM.U.u, DCM.U.u ])
  DCM     = euler_make_indices(DCM)
  *_, x   = euler_gen(DCM, DCM.Tp)
  DCM.U.u = DCM.U.u[:N, :]

  # Convolving neuronal signal with HRF
  for i in range(nr):
      tmp = np.fft.ifft(np.fft.fft(x[:,i]) * np.fft.fft(np.vstack([ h, np.zeros(( N*3 - len(h), 1 )) ])))
      y[:,i] = tmp[N:2*N];
  end

  # Sampling
  y = y[0:r_dt:, :]

  # Adding noise
  eps = np.random.randn(y.shape) * np.diag(np.std(y) / SNR)
  y_noise = y + eps

  # Saving the generated data
  DCM.Y.y = y_noise
  DCM.y = y
  DCM.x = x[N:2*N]

  return DCM
