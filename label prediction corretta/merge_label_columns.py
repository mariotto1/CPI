import scipy.io as sio
import glob
import numpy
import sys

target=sys.argv[1]+'/'

for file in glob.glob(target+'*.mat'):
    sim=sio.loadmat(file)["export_down"]
    sim[:,-2]=sim[:,-1]*2-sim[:,-2]
    sio.savemat(file, {'export_down': sim[:,:-1]})