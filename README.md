# ISAC_4D_IMaging
4D ISAC Imaging Simulation Based on Millimeter Wave OFDM Signals with MUSIC Algorithm Written in Matlab
# Document Structure
* 2D_FFT+2D_MUSCI
  * ref_ofdm_imaging_2DFFT_2DMUSIC.m  (Main function)
  * qamxxx.m & demoduqamxxx.m  (Modulation and demodulation)
  * xxxx_CFAR.m  (CFAR Detection)
  * environment_SE.m  (Simplified version of scatterer simulation)
  * environment.m  (Scatterers simulation)
  * environment_disp.m  (Display the simulation of environment)
  * goldseq.m & m_generate.m  (Sequence generation)
  * rcoswindow.m  (OFDM windowing algorithm)
## Imaging Results Presentation
![original](./2D_FFT_2D_MUSIC/image/original_environment.png)
![result](./2D_FFT_2D_MUSIC/image/2D_FFT+2D_music_result.png)

* 4D_FFT
  * ref_ofdm_imaging_4DFFT.m  (Main function)
  * qamxxx.m & demoduqamxxx.m  (Modulation and demodulation)
  * xxxx_CFAR.m  (CFAR Detection)
  * environment_SE.m  (Simplified version of scatterer simulation)
  * environment.m  (Scatterers simulation)
  * environment_disp.m  (Display the simulation of environment)
  * goldseq.m & m_generate.m  (Sequence generation)
  * rcoswindow.m  (OFDM windowing algorithm)
## Imaging Results Presentation
![original](./4D_FFT/image/original_environment.png)
![result](./4D_FFT/image/4DFFT_32_32RX_result.png)
