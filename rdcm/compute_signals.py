#
# Closely following code of TAPAS rDCM
#

import numpy as np
from .generate import generate


def compute_signals(DCM, output, options):
'''
Computes true and predicted signals

Input:
  DCM         - model structure
  output      - model inversion results
  options     - estimation options

Output:
  output      - model inversion results with model fits
'''

  # true or measured signal
  output.signal.y_source = DCM.Y.y.flatten()

  # true (deterministic) signal / VBL signal
  if options.type == 's':

      # if signal should be computed
      if options.compute_signal:

          # noise-free signal
          DCMs = generate(DCM, options, np.inf)
          output.signal.y_clean = DCMs.Y.y.flatten()

          # compute the MSE of the noisy data
          output.residuals.y_mse_clean = np.mean((output.signal.y_source - output.signal.y_clean)**2)

  else:

      # VBL predicted signal
      if hasattr(DCM,'y'):
          output.signal.y_pred_vl     = DCM.y.flatten()
          output.residuals.y_mse_vl   = np.mean((output.signal.y_source - output.signal.y_pred_vl)**2)
      else:
          output.residuals.y_mse_vl   = [];

  # store true or measured temporal derivative (in frequency domain)
  yd_source_fft = output.temp.yd_source_fft
  yd_source_fft[~np.isfinite(yd_source_fft)] = 0
  output.signal.yd_source_fft = yd_source_fft.flatten()

  # if signal in time domain should be computed
  if options.compute_signal:

      # get rDCM parameters
      DCM.Tp   = output.Ep

      # no baseline for simulated data
      if options.type == 's':
          DCM.Tp.baseline = np.zeros(( DCM.Tp.C.shape[0], 1 ));

      # increase sampling rate
      r_dt     = DCM.Y.dt/DCM.U.dt
      DCM.Y.dt = DCM.U.dt

      # posterior probability (for sparse rDCM)
      if hasattr(output,'Ip'):
          DCM.Tp.A = DCM.Tp.A * output.Ip.A
          DCM.Tp.C = DCM.Tp.C * output.Ip.C
    
      # generate predicted signal (tapas_rdcm_generate)
      DCM_rDCM = generate(DCM, options, np.inf)

      # add the confounds to predicted time series
      for t in range(DCM.Y.y.shape[0]):
          for t in range(DCM.Y.y.shape[1]):
              DCM_rDCM.Y.y[t,r] = DCM_rDCM.Y.y[t,r] + DCM_rDCM.Tp.baseline[r,:] @ DCM_rDCM.U.X0[r_dt*t,:].T

      # turn into vector
      output.signal.y_pred_rdcm = DCM_rDCM.Y.y.flatten()


  # store predicted temporal derivative (in frequency domain)
  yd_pred_rdcm_fft                = output.temp.yd_pred_rdcm_fft
  output.signal.yd_pred_rdcm_fft  = yd_pred_rdcm_fft.flatten()

  # remove the temp structure
  delattr(output,'temp') 


  # asign the region names
  if hasttr(DCM.Y,'name'):
      output.signal.name = DCM.Y.name
  
  # compute the MSE of predicted signal
  if options.compute_signal:
      output.residuals.y_mse_rdcm     = np.mean((output.signal.y_source - output.signal.y_pred_rdcm)**2)
      output.residuals.R_rdcm         = output.signal.y_source - output.signal.y_pred_rdcm
  
  # store the driving inputs
  output.inputs.u     = DCM.U.u;
  output.inputs.name  = DCM.U.name;
  output.inputs.dt    = DCM.U.dt;

  return output
