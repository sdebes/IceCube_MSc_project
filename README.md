# MSc_Thesis_IceCube
This repo contains my Master's thesis and the accompanying code.

Short resume of my thesis:
# Intro

IceCube is the world's largest neutrino telescope consisting of 5160 light-detecting Digital Optical Modules (DOMs) deposited into a cubic kilometer of ice 1.5km below the South Pole. Neutrinos are hard to detect and IceCube detects orders of magnitude more muons than neutrinos. Because we have so much more data on muons, they are better suited for high statistics detector calibration. This project highlights the data and simulation (MC) agreement of muons in IceCube in an effort to reduce systematic errors in other IceCube analyses, such as neutrino oscillation analyses and the study of dark matter.

# PulseMerger
A large sample of stopped muons is selected using the graph neural network (GNN) model DynEdge from the GraphNeT python library, and a pulse-merging algorithm called PulseMerger is developed to merge pulses that are believed to be incorrectly split in the data recording process. PulseMerger is shown to improve the agreement between the charge distributions of data and MC in hard local coincidence hits, and does not affect soft local coincidence hits.

# Photon track reconstruction
The muons emit light at a constant angle of around 49 degrees (90 degrees minus the Cherenkov angle), making it possible to calculate when and where light (or photons) is emitted.
For each DOM activated or not, in each event the vector of the recorded or un-recorded photon is reconstructed. From these reconstructed photon vectors, the following variables are extracted:

  Photon distance

  Photon azimuth angle
  
  Photon zenith angle
  
  Photon z (depth of the halfway point between emission and detection of the light)

# Efficiency and timing of the detector
With these variables, it is possible to determine the efficiency of the detector as a function of distance, zenith angle, azimuth angle, and depth.

Because we know when and where the muon and the emitted photons are, we can calculate backward from detection to determine the time at which the muon stopped within the detector, called tA.
The variation of tA within one event indicates the detector's resolution and timing and reveals how much light scatters in the ice. The resolution is measured at different distances (5m intervals) to the DOMs and is fitted with a linear function, revealing the contribution of the ice to the resolution. The process is done for predicted and true values of MC, and the difference between the two shows how much the reconstruction affects the resolution, which we can only assume is similar for recorded data, where no true values exist.
