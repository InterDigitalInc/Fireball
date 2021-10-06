from fireball import Model, DashOptions, myPrint

# ************************************************************************************************
options = DashOptions(
    [
     ('help',     'Show this help',                 ['bool']),
     ('h',        None,                             ['bool']),
     
     ('in',           'Input Model Parameters. (An "npz" file exported by a Fireball model)',   ['string', 'required']),
     ('out',          'Output numpy file.',                                                     ['string', 'required']),
     ('qInfo',        'Comma separated quantization info: minBits,maxBits,MSE,pdf',             ['string', 'default:2,8,.0001,.8']),
    ])

# ************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(1)

    qbitsStr = options.get('qInfo')
    qbitsParts = qbitsStr.split(',')
    if len(qbitsParts) != 4:
        myPrint("Invalid qbits format. It should comma separated info. (minBits,maxBits,MSE,pdf)", color="red")
        exit(1)

    Model.quantizeModel(options.get('in'), options.get('out'),
                        minBits=int(qbitsParts[0]), maxBits=int(qbitsParts[1]), mseUb=float(qbitsParts[2]),
                        pdfFactor=float(qbitsParts[3]),
                        gradsInfo=None, quiet=False)
