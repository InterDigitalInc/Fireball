import time
import numpy as np

from fireball import Model, DashOptions, myPrint
from fireball.layers import Layers

# ************************************************************************************************
options = DashOptions(
    [
        ('help',        'Show this help',                                           ['bool']),
        ('h',           None,                                                       ['bool']),
     
        ('in',          'Input Model.',                                             ['string', 'required']),
        ('out',         'Output model (With specified layer converted to LR/LDR).', ['string', 'required']),
        ('layers',      'The layers to apply LR (Comma separated)',                 ['string', 'required']),
        ('mse',         'The MSE for the low-rank decomposition.',                  ['float', 'default:0.005']),
    ])

# ************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(1)

    model = Model.makeFromFile(options.get('in'), gpus="-1")

    model.printLayersInfo()
    model.initSession()
#    Layers.printAllParamInfo(model.layers)

    mse = options.get('mse')
    layers = options.get('layers').split(',')
    layerParams = [ (layer, mse) for layer in layers]
    
    myPrint('\nNow reducing number of network parameters ... ')
    t0 = time.time()
    model.createLrModel(options.get('out'), layerParams)
    myPrint('Done. (%.2f Seconds)'%(time.time()-t0))
