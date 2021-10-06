# Copyright (c) 2019-2020 InterDigital AI Research Lab
"""
This file contains the implementation of LDR functionality for Fireball Models.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 04/03/2019    Shahab Hamidi-Rad       Created the file.
# 08/25/2020    Shahab                  Started documenting version history.
# **********************************************************************************************************************
import numpy as np
import tensorflow as tf

TFCX_TYPE = tf.complex128
TFFP_TYPE = tf.float64

# **********************************************************************************************************************
# python 2/3 compatibility (Remove when dropping support for python2.7)
try:                range = xrange              # Python 2
except NameError:   pass                        # Python 3

# **********************************************************************************************************************
def getRootPowers(x, n):
    if x>=0:    r1 = x**(1./n)
    else:       r1 = ((-x)**(1./n))*np.exp(np.pi*1j/n)
    rp = [1.0]
    for _ in range(n-1): rp += [rp[-1]*r1]
    return np.array(rp)

# **********************************************************************************************************************
def getNthRootsOf(x, n):
    v = n*[0.0]
    v[1] = 1 if x>=0 else np.exp(np.pi*(0+1j)/n)
    return (np.abs(x)**(1./n))*np.fft.fft(np.array(v))

# **********************************************************************************************************************
def getCirculant(f0, v, cl):
    mm = v.shape[1]

    fi = 1
    vf = np.flip(v.copy(),1)
    c = vf.copy()
    while c.shape[1]<cl:
        fi *= f0
        c = np.append(c, vf*fi, 1)
    return np.roll( c[:,:cl], 1-mm, 1)

# **********************************************************************************************************************
def getCTfmn(f, n, v):
    # v: rxm   (n>m)
    # returns: rx(m+n-1)
    m = tf.shape(v)[1]
    cLen = m+n
    k = n//m + 2
    jv = tf.reverse(v,[-1])
    c2d = tf.reshape(tf.tile(tf.expand_dims(tf.cast(jv, TFFP_TYPE), 0), [1,1,k]), [-1,k,m])*tf.reshape(f**tf.cast(tf.range(k), TFFP_TYPE),[k,1])
    return tf.cast( tf.concat((tf.slice( tf.reshape(c2d, [-1,k*m]), [0,m-1], [-1,cLen-m+1]),
                               tf.slice( tf.reshape(c2d, [-1,k*m]), [0,0], [-1,m-1])), -1), TFCX_TYPE)

## **********************************************************************************************************************
#def getCTfmn2(f, n, v):
#    # v: rxm   (n>m)
#    # returns: rx(m+n-1)
#    m = tf.shape(v)[1]
#    cLen = m+n
#    k = n//m + 2
#    jv = tf.reverse(v,[-1])
#    c2d = tf.reshape(tf.tile(tf.expand_dims(tf.cast(jv, TFFP_TYPE), 0), [1,1,k]), [-1,k,m])*tf.reshape(f**tf.cast(tf.range(k), TFFP_TYPE),[k,1])
#    return tf.cast( tf.roll( tf.slice( tf.reshape(c2d, [-1,k*m]), [0,0], [-1,cLen]), 1-m, -1), TFCX_TYPE)

# **********************************************************************************************************************
def getZf(n,f):
    zf = np.roll(np.identity(n),1,0)
    zf[0,-1] = f
    return zf

# **********************************************************************************************************************
def getSteingDisp(m, e, f):     # Delta
    return m - np.matmul( np.matmul(getZf(m.shape[0], e), m), getZf(m.shape[1], f))

# ******************************************************************************************************************
def jx(x):                  return tf.reverse(x,[-1])
def crop(x,numCols):        return tf.slice(x, [0,0,0], [-1,-1,numCols])
def append0s(x,num0s):
    xShape = tf.shape(x)
    return tf.concat((x, tf.zeros((xShape[0],xShape[1],num0s), dtype=TFCX_TYPE)),-1)

# **********************************************************************************************************************
def getEq1InferenceGraph(x, g, jh, e, f, m, n, r):  # (e=0 and f!=0) or m<n
    b = tf.shape(x)[0]     # Batch Size
    x0 = tf.cast( tf.reshape(x, [b, 1, n]), TFCX_TYPE)                                  # (b, 1, n)
    if e!=0:
        # Note that f cannot be 0 here because Eq2 is used in that case.
        pf = getRootPowers(f, n)                                                        # (n,)
        rf = getNthRootsOf(f, n)                                                        # (n,)
        oneSuberfm = 1.-e*(rf**m)                                                       # (n,)
        # For square cases, this can be represented by just an scaler
        if m==n:  oneSuberfm = oneSuberfm[0]
        if np.any(np.abs(oneSuberfm)<.000001):
            raise NotImplementedError("e/f combination (%d,%d) does not work with matrix dimensions (%dx%d)"%(e, f, m, n))
        oneSuberfm = tf.constant(oneSuberfm, TFCX_TYPE, [1,1,n])                        # (1,1,n)

        if f==1:    x0 = tf.ifft(tf.fft(x0)/oneSuberfm)                                 # (b, 1, n)
        else:       x0 = tf.ifft(tf.fft(pf*x0)/oneSuberfm)/pf                           # (b, 1, n)

    if f==0 or m>n:
        # Use inefficient method for Zf,n,m(Jhi)T only if f is 0 or m>n
        omegaCf4JH = tf.fft( getCTfmn(f, m, jh) )                                       # (r, m+n)
        x1 = tf.fft( append0s( jx(x0), m) )                                             # (b, 1, m+n)
        x2 =  crop( tf.ifft( x1*omegaCf4JH), m)                                         # (b, r, m)
    elif f==1:
        omegaPfJH = tf.fft(tf.cast(jh, TFCX_TYPE))                                      # (r, n)
        x1 = tf.ifft( jx(x0) )                                                          # (b, 1, n)
        x2 = crop( tf.fft(x1*omegaPfJH), m)                                             # (b, r, m)
    else:
        pf = getRootPowers(f, n)                                                        # (n,)
        omegaPfJH = tf.fft(tf.cast(jh, TFCX_TYPE)*pf)                                   # (r, n)
        x1 = tf.ifft( jx(x0)/pf)                                                        # (b, 1, n)
        x2 = crop( pf*tf.fft(x1*omegaPfJH), m)                                          # (b, r, m)

    if e==1:
        omegaPeG = tf.fft(tf.cast(g, TFCX_TYPE))                                        # (r, m)
        x3 = omegaPeG * tf.fft(x2)                                                      # (b, r, m)
        x4 = tf.ifft( tf.reduce_sum(x3, axis=1) )                                       # (b, m)

    elif e!=0:
        pe = getRootPowers(e, m)                                                        # (m,)
        omegaPeG = tf.fft(tf.cast(g, TFCX_TYPE)*pe)                                     # (r, m)
        x3 = omegaPeG * tf.fft(pe*x2)                                                   # (b, r, m)
        x4 = tf.ifft( tf.reduce_sum(x3, axis=1))/pe                                     # (b, m)

    else:
        # Use inefficient method for Ze(gi) only if e is 0
        omegaCe4G  = tf.fft( getCTfmn(e, m, g) )                                        # (r, 2m)
        x3 = crop( tf.fft( omegaCe4G * tf.ifft( append0s(x2, m))), m)                   # (b, r, m)
        x4 = tf.reduce_sum(x3, axis=1)                                                  # (b, m)

    return tf.reshape( tf.real(x4), [b,m])                                              # (b, m)

# **********************************************************************************************************************
def getEq2InferenceGraph(x, g, jh, e, f, m, n, r):  # (e!=0 and f==0) or m>n
    b = tf.shape(x)[0]     # Batch Size
    x0 = tf.cast( tf.reshape(x, [b, 1, n]), TFCX_TYPE)                                  # (b, 1, n)
    if f==0:
        # Use inefficient method for Zf(Jhi) only if f is 0
        omegaCf4JH = tf.fft( getCTfmn(f, n, jh) )                                       # (r, m+n)
        x1 = tf.fft( append0s( jx(x0), n) )                                             # (b, 1, 2n)
        x2 = crop( tf.ifft( x1*omegaCf4JH), n)                                          # (b, r, n)
    elif f==1:
        omegaPfJH = tf.fft(tf.cast(jh, TFCX_TYPE))                                      # (r, n)
        x1 = tf.ifft( jx(x0) )                                                          # (b, 1, n)
        x2 = tf.fft( omegaPfJH * x1)                                                    # (b, r, n)
    else:
        pf = getRootPowers(f, n)                                                        # (n,)
        omegaPfJH = tf.fft(tf.cast(jh, TFCX_TYPE)*pf)                                   # (r, n)
        x1 = tf.ifft( tf.cast( jx(x0), TFCX_TYPE)/pf )                                  # (b, 1, n)
        x2 = pf * tf.fft( omegaPfJH * x1)                                               # (b, r, n)

    # Use inefficient method for Ze,m,n(gi) only if e is 0 or m<n
    if e==0:
        # Note that f cannot be nonzero here because Eq1 is used in that case.
        omegaCe4G  = tf.fft( getCTfmn(e, n, g) )                                        # (r, 2n)
        x3 = crop( tf.fft( omegaCe4G * tf.ifft( append0s(x2, m))), m)                   # (b, r, m)
        x4 = tf.reduce_sum(x3, axis=1)                                                  # (b, 1, m)

    elif m<n:
        # we know f=0 and e!=0
        omegaCe4G  = tf.fft( getCTfmn(e, n, g) )                                        # (r, 2n)
        x3 = crop( tf.fft( omegaCe4G * tf.ifft(append0s(x2, m))), m)                    # (b, r, m)
        x4 = tf.reduce_sum(x3, axis=1)                                                  # (b, 1, m)

    elif e==1:
        omegaPeG = tf.fft(tf.cast(g, TFCX_TYPE))                                        # (r, m)
        x3 = omegaPeG * tf.fft( append0s(x2, m-n))                                      # (b, r, m)
        if f==0:    x4 = tf.ifft( tf.reduce_sum(x3, axis=1) )                           # (b, 1, m)
        else:
            re = getNthRootsOf(e, m)                                                    # (m,)
            oneSubfren = 1.-f*(re**n)                                                   # (m,)
            # For square cases, this can be represented by just an scaler
            if m==n:  oneSubfren = oneSubfren[0]
            if np.any(np.abs(oneSubfren)<.000001):
                raise NotImplementedError("e/f combination (%d,%d) does not work with matrix dimensions (%dx%d)"%(e, f, m, n))
            x4 = tf.ifft( tf.reduce_sum(x3, axis=1)/oneSubfren)                         # (b, 1, m)

    else:
        pe = getRootPowers(e, m)                                                        # (m,)
        omegaPeG = tf.fft(tf.cast(g, TFCX_TYPE)*pe)                                     # (r, m)
        x3 = omegaPeG * tf.fft( pe * append0s(x2, m-n))                                 # (b, r, m)
        if f==0:    x4 = tf.ifft(tf.reduce_sum(x3, axis=1))/pe                          # (b, 1, m)
        else:
            re = getNthRootsOf(e, m)                                                    # (m,)
            oneSubfren = 1.-f*(re**n)                                                   # (m,)
            # For square cases, this can be represented by just an scaler
            if m==n:  oneSubfren = oneSubfren[0]
            if np.any(np.abs(oneSubfren)<.000001):
                raise NotImplementedError("e/f combination (%d,%d) does not work with matrix dimensions (%dx%d)"%(e, f, m, n))
            x4 = tf.ifft(tf.reduce_sum(x3, axis=1)/oneSubfren)/pe                       # (b, 1, m)

    return tf.reshape( tf.real(x4), [b,m])                                              # (b, m)

# **********************************************************************************************************************
def getZ(v,f,n=None):    # returns Zf,m,n(v)
    m = v.shape[0]
    if n is None: n=m
    z = np.zeros((m,n), dtype=v.dtype)
    for r in range(m):
        for c in range(n):
            if r>=c:
                z[r,c] = v[r-c]
                continue
            k = (c-r-1)//m + 1
            l = (c-r-1)%m
            z[r,c] = (f**k)*v[m-l-1]
    return z

# **********************************************************************************************************************
def tfGetZfmnVdotX(f, m, n, v, x):
    # x: bxn    (type must be TFFP_TYPE)
    # v: rxm    (type must be TFFP_TYPE)
    # output: rxbxm
    k = (n-1)//m + 1
    vv = tf.tile( tf.expand_dims(v, 0), [1,1,k+1] )
    vv = tf.reshape( vv, [-1,k+1,m] )
    vv *= np.reshape(f**np.arange(k,-1,-1), (1,k+1,1))
    vv = tf.reshape(vv, (-1,(k+1)*m))
    vv = tf.slice(vv, [0,k*m-n+1], [-1, -1])

    input = tf.reshape(vv, (-1,m+n-1,1))
    filters = tf.reshape(tf.transpose(tf.reverse(x,[-1])),[n,1,-1])
    return tf.transpose(tf.nn.conv1d(input, filters,1, 'VALID'), perm=[0, 2, 1])

## Testing tfGetZfmnVdotX:
#m,n = 5,4
#e,f = 2,2
#r = 3
#b = 2
#
#tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
#                          intra_op_parallelism_threads=8,
#                          gpu_options=tf.GPUOptions(allow_growth=True))
#session = tf.Session(config=tfConfig)
#session.run( tf.global_variables_initializer() )
#
#x = np.float64(np.reshape(range(b*n), (b,n)))+1
#v = np.float64(np.reshape(range(r*m), (r,m)))+1
#print('v:\n'+str(v))
#print('x:\n'+str(x))
#
#out = tfGetZfmnVdotX(f,m,n,v,x)
#print('out:\n'+str(session.run(   out   )))
#for bb in range(b):
#    print( 'Batch %d'%(bb) )
#    for rr in range(r):
#        print('    Zfmn(V[%d]).X[%d]: '%(rr,rr)+str(getZ(v[rr],f,n).dot(x[bb])))
#exit(0)

# **********************************************************************************************************************
def tfGetZfmnVTdotX(f, m, n, v, x):
    # x: bxn    (type must be TFFP_TYPE)
    # v: rxn    (type must be TFFP_TYPE)
    # output: rxbxm
    k = (n-1)//m + 1
    vv = tf.tile( tf.expand_dims(v, 0), [1,1,k+1] )
    vv = tf.reshape( vv, [-1,k+1,m] )                                       # (r, (k+1), m)
    vv *= np.reshape(f**np.arange(k,-1,-1), (1,k+1,1))
    vv = tf.reshape(vv, (-1,(k+1)*m))                                       # (r, (k+1)*m)
    vv = tf.slice(vv, [0,k*m-n+1], [-1, -1])

    input = tf.reshape(vv, (-1,m+n-1,1))                                    # (r, m+n-1, 1)
    filters = tf.reshape(tf.transpose(x),[m,1,-1])                          # (n, 1, b)
    return tf.reverse( tf.transpose(tf.nn.conv1d(input, filters,1,'VALID'), perm=[0, 2, 1]), [-1])  # (r, b, m)

## Testing tfGetZfmnVTdotX:
#m,n = 5,4
#e,f = 2,2
#r = 3
#b = 2
#
#tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
#                          intra_op_parallelism_threads=8,
#                          gpu_options=tf.GPUOptions(allow_growth=True))
#session = tf.Session(config=tfConfig)
#session.run( tf.global_variables_initializer() )
#
#x = np.float64(np.reshape(range(b*n), (b,n)))+1
#v = np.float64(np.reshape(range(r*n), (r,n)))+1
#print('v:\n'+str(v))
#print('x:\n'+str(x))
#
#out = tfGetZfmnVTdotX(f,n,m,v,x)
#print('out:\n'+str(session.run(   out   )))
#for bb in range(b):
#    print( 'Batch %d'%(bb) )
#    for rr in range(r):
#        print('    Zfmn(V[%d]).X[%d]: '%(rr,rr)+str(np.transpose(getZ(v[rr],f,m)).dot(x[bb])))
#exit(0)

# **********************************************************************************************************************
def tfGetZfmnVdotXr(f, m, n, v, x, r):
    # x: rxbxn  (type must be TFFP_TYPE)
    # v: rxm    (type must be TFFP_TYPE)
    # output: rxbxm
    k = (n-1)//m + 1
    vv = tf.tile( tf.expand_dims(v, 0), [1,1,k+1] )
    vv = tf.reshape( vv, [-1,k+1,m] )
    vv *= np.reshape(f**np.arange(k,-1,-1), (1,k+1,1))
    vv = tf.reshape(vv, (-1,(k+1)*m))
    vv = tf.slice(vv, [0,k*m-n+1], [-1, -1])

    inputs = tf.reshape(vv, (r,1,m+n-1,1))
    filters = tf.reshape(tf.transpose(tf.reverse(x,[-1]), perm=[0, 2, 1]),[r, n, 1, -1])
    return tf.transpose(tf.reshape(tf.map_fn(lambda rr: tf.nn.conv1d(inputs[rr], filters[rr], 1, 'VALID'),
                                             tf.range(r),
                                             dtype=TFFP_TYPE),
                                   (r,m,-1)),
                        perm=[0, 2, 1])

## Testing tfGetZfmnVdotXr
#m,n = 5,4
#e,f = 2,2
#r = 3
#b = 2
#tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
#                          intra_op_parallelism_threads=8,
#                          gpu_options=tf.GPUOptions(allow_growth=True))
#session = tf.Session(config=tfConfig)
#session.run( tf.global_variables_initializer() )
#
#x = np.float64(np.reshape(range(b*r*n), (r,b,n)))+1
#v = np.float64(np.reshape(range(r*m), (r,m)))+1
#print('v:\n'+str(v))
#print('x:\n'+str(x))
#
#out = session.run( tfGetZfmnVdotXr(f, m, n, v, x, r) )
#print('out:\n'+str(out))
#print(out.shape)
#
#for bb in range(b):
#    print( 'Batch %d'%(bb) )
#    for rr in range(r):
#        print('    Zfmn(V[%d]).X[%d]: '%(rr,rr)+str(getZ(v[rr],f,n).dot(x[rr,bb])))
#exit(0)

# **********************************************************************************************************************
def getEq1TrainingGraph(x, g, jh, e, f, m, n, r):   # (e=0 and f!=0) or m<n
    if e!=0:
        rf = getNthRootsOf(f, n)                                    # (n,)
        oneSuberfm = 1.-e*(rf**m)                                   # (n,)
        if (e!=0) and np.any(np.abs(oneSuberfm)<.000001):
            raise NotImplementedError("e/f combination (%d,%d) does not work for with matrix dimensions (%dx%d)"%(e, f, m, n))
        oneSuberfm = tf.constant(oneSuberfm, TFCX_TYPE, [1,n])      # (1,n)

        if f==1:
            x0 = tf.cast( tf.ifft(tf.fft( tf.cast( x, TFCX_TYPE) )/oneSuberfm), TFFP_TYPE)
        else:
            pf = tf.constant(getRootPowers(f, n), TFCX_TYPE, [1,n])     # (1,n)
            x0 = tf.cast( tf.ifft(tf.fft( pf * tf.cast( x, TFCX_TYPE) )/oneSuberfm)/pf, TFFP_TYPE)
    else:
        x0 = x

    jx = tf.reverse(x0,[-1])
    x1 = tfGetZfmnVTdotX(f, n, m, jh, jx)
    x2 = tfGetZfmnVdotXr(e, m, m, g, x1, r)
    return tf.reduce_sum(x2,0)

# **********************************************************************************************************************
def getEq2TrainingGraph(x, g, jh, e, f, m, n, r):   # (e!=0 and f==0) or m>n
    jx = tf.reverse(x,[-1])
    x1 = tfGetZfmnVTdotX(f, n, n, jh, jx)
    x2 = tfGetZfmnVdotXr(e, m, n, g, x1, r)

    if f==0:    return tf.reduce_sum(x2,0)

    re = getNthRootsOf(e, m)                                    # (m,)
    oneSubfren = 1.-f*(re**n)
    if (f!=0) and np.any(np.abs(oneSubfren)<.000001):
        raise NotImplementedError("e/f combination (%d,%d) does not work with matrix dimensions (%dx%d)"%(e, f, m, n))

    oneSubfren = tf.constant(oneSubfren, TFCX_TYPE, [1,m])      # (1,m)
    if e==1:    return tf.cast( tf.ifft(tf.fft( tf.cast( tf.reduce_sum(x2,0), TFCX_TYPE) )/oneSubfren), TFFP_TYPE)

    pe = tf.constant(getRootPowers(e, m), TFCX_TYPE, [1,m])     # (1,m)
    return tf.cast( tf.ifft(tf.fft( pe * tf.cast( tf.reduce_sum(x2,0), TFCX_TYPE) )/oneSubfren)/pe, TFFP_TYPE)

# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
def getLdrGraph(training, input, g, jh, e, f, m, n, r, dtype=tf.float32):
    # This function returns a subgraph which is an LDR replacement for a large matrix multiplication.
    g = tf.cast(g, TFFP_TYPE)
    jh = tf.cast(jh, TFFP_TYPE)
    input = tf.cast(input, TFFP_TYPE)
    if training:
        if e==0 and f!=0:   return tf.cast(getEq1TrainingGraph(input, g, jh, e, f, m, n, r), dtype)
        if e!=0 and f==0:   return tf.cast(getEq2TrainingGraph(input, g, jh, e, f, m, n, r), dtype)
        if m<=n:            return tf.cast(getEq1TrainingGraph(input, g, jh, e, f, m, n, r), dtype)
        return tf.cast(getEq2TrainingGraph(input, g, jh, e, f, m, n, r), dtype)
    else:
        if e==0 and f!=0:   return tf.cast(getEq1InferenceGraph(input, g, jh, e, f, m, n, r), dtype)
        if e!=0 and f==0:   return tf.cast(getEq2InferenceGraph(input, g, jh, e, f, m, n, r), dtype)
        if m<=n:            return tf.cast(getEq1InferenceGraph(input, g, jh, e, f, m, n, r), dtype)
        return tf.cast(getEq2InferenceGraph(input, g, jh, e, f, m, n, r), dtype)

## Testing getLdrGraph
#m,n = 5,4
#e,f = 2,2
#r=4
#b=2
#
#w = np.float32(np.random.randn(m,n))
#x = np.float32(np.random.randn(b,n))
#
#wx0 = np.matmul(w,x[0])
#wx1 = np.matmul(w,x[1])
#
#lm = getSteingDisp(w, e, f)
#u, s, vT = np.linalg.svd(lm, full_matrices=True)
#g = np.transpose(u[:,:r])                           # (r,m)
#jh = np.flip( np.diag(s[:r]).dot(vT[:r,:]),-1)      # (r,n)
#
#
#tf.reset_default_graph()
#tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
#                          intra_op_parallelism_threads=8,
#                          gpu_options=tf.GPUOptions(allow_growth=True))
#session = tf.Session(config=tfConfig)
#session.run( tf.global_variables_initializer() )
#
#input = tf.placeholder( TFFP_TYPE, shape=[None,n], name='input' )
#output = getLdrGraph(False, input, g, jh, e, f, m, n, r)
#wx = session.run( output, feed_dict={input:x})
#print(wx)
#print(x.dot(np.transpose(w)))
#exit(0)

# **********************************************************************************************************************
def testLdrGraph(training):
    # Unit-testing the getLdrGraph function:
    success = 0
    tests = 0
    notSupported = 0
    b=2

    for m in [4, 5, 22, 31, 35]:
        for n in [4, 5, 22, 31, 35]:
            for e in [0, 1, 2, 3, -1, -2, -3]:
                for f in [0, 1, 2, 3, -1, -2, -3]:
                    tests += 1
                    r = min((m,n))
                    w = np.random.randn(m,n)
                    x = np.random.randn(b,n)

                    wx0 = np.matmul(w,x[0])
                    wx1 = np.matmul(w,x[1])
                    
                    lm = getSteingDisp(w, e, f)
                    u, s, vT = np.linalg.svd(lm, full_matrices=True)
                    g = np.transpose(u[:,:r])                           # (r,m)
                    jh = np.flip( np.diag(s[:r]).dot(vT[:r,:]),-1)      # (r,n)

                    try:
                        tf.reset_default_graph()

                        tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
                                                  intra_op_parallelism_threads=8,
                                                  gpu_options=tf.GPUOptions(allow_growth=True))
                        session = tf.Session(config=tfConfig)
                        session.run( tf.global_variables_initializer() )

                        input = tf.placeholder( TFFP_TYPE, shape=[None,n], name='input' )
                        g = tf.constant(g, TFFP_TYPE)
                        jh = tf.constant(jh, TFFP_TYPE)
                        output = getLdrGraph(training, input, g, jh, e, f, m, n, r)
                        wx = session.run( output, feed_dict={input:x})

                        if np.abs(wx[0]-wx0).sum()>.1 or np.abs(wx[1]-wx1).sum()>.1:
                            print('m,n = %d,%d     e,f = %d,%d => Error!!!!!!!!!!!!: %f,%f'%(m,n,e,f,
                                                                                             np.abs(wx[0]-wx0).sum(),
                                                                                             np.abs(wx[1]-wx1).sum()))
                            continue

                        success += 1
                    except NotImplementedError as err:
                        print('m,n = %d,%d     e,f = %d,%d => "%s"'%(m,n,e,f,str(err)))
                        if str(err)[:15]=='e/f combination':    notSupported+=1

    print('Tested %d cases, %d succeeded, %d failed, %d not supported!'%(tests, success,
                                                                         tests-success-notSupported,notSupported))

## **********************************************************************************************************************
def profileLdrGraph(training):
    # Profiling getLdrGraph calls:
    import time
    b = 2
    timeLdr = 0
    timeMatmul = 0
    print('e,f,m,n,r,b,LdrTime,matmulTime')
    for m in [1024, 8192, 16384]:
        for n in [1024, 8192, 16384]:
            for e in [-2,-1,0,1,2]:
                for f in [-2,-1,0,1,2]:
                    for r in [10, 50]:
                        g = np.random.randn(r,m)
                        jh = np.random.randn(r,n)
                        x = np.random.randn(b,n)

                        tf.reset_default_graph()
                        try:
                            input = tf.placeholder( TFFP_TYPE, shape=[None,n], name='input' )
                            g = tf.constant(g, TFFP_TYPE)
                            jh = tf.constant(jh, TFFP_TYPE)
                            output = getLdrGraph(training, input, g, jh, e, f, m, n, r)
                        except NotImplementedError as err:
                            if str(err)[:15]=='e/f combination':
                                continue

                        tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
                                                  intra_op_parallelism_threads=8,
                                                  gpu_options=tf.GPUOptions(allow_growth=True))
                        session = tf.Session(config=tfConfig)
                        session.run( tf.global_variables_initializer() )

                        ww = np.random.randn(b,m,n)
                        xx = np.reshape(x, [b,n,1])
                        t1 = time.time()

                        for _ in range(10):
                            wx = session.run( output, feed_dict={input:x})

                        t2 = time.time()

                        for _ in range(10):    wwx = ww.dot(xx)

                        t3 = time.time()
                        timeLdr += t2-t1
                        timeMatmul += t3-t2
                        print('%d,%d,%d,%d,%d,%d,%f,%f'%(e, f, m, n, r, b, (t2-t1), (t3-t2)))

# **********************************************************************************************************************
def getLdrGH(w, e, f, r, wGrads=None, dtype=TFFP_TYPE):
    if dtype in [np.float32, tf.float32]:
        npType = np.float32
        tfType = tf.float32
    else:
        npType = np.float64
        tfType = tf.float64

    print( '\nRunning SVD to get initial values for G and JH...')
    lm = getSteingDisp(np.transpose(w), e, f)
    u, s, vT = np.linalg.svd(lm, full_matrices=False)
    ssr = np.sqrt(s[:r])
    g = (npType)(np.transpose(u[:,:r]*ssr))                         # rxm
    jh = (npType)(np.flip( np.diag(ssr).dot(vT[:r,:]),-1))          # rxn


    n,m = w.shape
    batchSize = 64
    numEpochs = 10     # Set to zero to bypass the optimization below and use just the results of SVD
    numEpochs = 0

    learningRateInit = .001
    learningRateMin = .00005
    LEARNING_DECAY_RATE = .999
    learningDecayStep = (((n//batchSize)*numEpochs)//
                         (np.log(learningRateMin/learningRateInit)/np.log(LEARNING_DECAY_RATE)))

    SEED = 1234
    ldrGraph = tf.Graph()
    with ldrGraph.as_default():
        tfIdIndexes = tf.placeholder( tf.int32, shape=[None,1], name='IdIndexes' )
        tfIdColumns = tf.reshape(tf.one_hot(tfIdIndexes, n, dtype=tfType), [-1,n])
        tfWColumns = tf.placeholder( tfType, shape=[None,m], name='WColumns' )
        if wGrads is not None:
            tfWGrads = tf.placeholder( tfType, shape=[None,m], name='WGrads' )
        
        tfBatch = tf.Variable(0,name='BatchCounter')   # Incremented once per batch
        tfLearningRate = tf.train.exponential_decay(learningRateInit,       # Base learning rate.
                                                    tfBatch,                # Current index into the dataset.
                                                    learningDecayStep,      # Decay step.
                                                    LEARNING_DECAY_RATE,    # Decay rate.
                                                    staircase=True)
                                 
        tfG = tf.Variable(g, dtype=tfType, name='G')
        tfJH = tf.Variable(jh, dtype=tfType, name='JH')

        tfWdotIdColumns = getLdrGraph(False, tfIdColumns, tfG, tfJH, e, f, m , n, r, tfType)
        if wGrads is None:
            tfLossB4Reg = tf.reduce_mean(tf.square(tfWColumns - tfWdotIdColumns))
            tfLoss = tfLossB4Reg #+ .01*tf.reduce_mean(tf.square(tfG)) + .01*tf.reduce_mean(tf.square(tfJH))
        else:
#            tfLoss = tf.reduce_sum(tf.square(tfWColumns - tfWdotIdColumns)*tfWGrads)/tf.reduce_sum(tfWGrads)
            tfLoss = tf.reduce_mean(tf.square(tfWColumns - tfWdotIdColumns)*tfWGrads)
            tfLossB4Reg = tfLoss

        tfOpt = tf.train.AdamOptimizer(learningRateInit).minimize(tfLoss, global_step=tfBatch)

    tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=tfConfig, graph=ldrGraph)
    session.run( [v.initializer for v in ldrGraph.get_collection( tf.GraphKeys.GLOBAL_VARIABLES)] )

    allIdx = np.arange(n)
    # Calculating initial error:
    a = 0
    sumLoss = 0.0
    while a < n:
        idx = allIdx[a:a+batchSize]
        feedDic={ tfIdIndexes: idx.reshape([-1,1]), tfWColumns: (npType)(w[idx])}
        if wGrads is not None:  feedDic[tfWGrads] = (npType)(wGrads[idx])
        sumLoss += session.run(tfLoss, feed_dict=feedDic)
        a += batchSize
    avgLoss = sumLoss*batchSize/n
    print('MSE After SVD: %.9f'%(avgLoss))

    if numEpochs==0:
        # No SGD fine-tuning:
        session.close()
        return g,jh,e,f,r,avgLoss

    # Now Start Training:
    print('Now optimizing for best G, JH%s...'%('' if wGrads is None else ' (Using Weight Gradients) '))
    print('e,f=%f,%f - r=%d, Num Epoch: %d, BatchSize: %d'%(e,f,r,numEpochs,batchSize))
    
    minLoss = 0
    for ep in range(numEpochs):
#        np.random.shuffle(allIdx)
        a = 0
        sumLoss = 0.0
        while a < n:
            idx = allIdx[a:a+batchSize]
            feedDic={ tfIdIndexes: idx.reshape([-1,1]), tfWColumns: (npType)(w[idx])}
            if wGrads is not None:  feedDic[tfWGrads] = (npType)(wGrads[idx])
            _,loss = session.run([tfOpt, tfLoss], feed_dict=feedDic)
            sumLoss += loss*len(idx)
            a += len(idx)
        avgLoss = sumLoss/n
        print('  Epoch %d: Avg Loss: %.9f'%(ep+1, avgLoss))
        if minLoss==0:
            minLoss = avgLoss
            g,jh = session.run([tfG, tfJH])
        elif avgLoss<minLoss:
            minLoss = avgLoss
            g,jh = session.run([tfG, tfJH])

    print('Final MSE: %.9f'%(avgLoss))
    print('Min MSE: %.9f'%(minLoss))

    session.close()
    return g,jh,e,f,r,minLoss

# **********************************************************************************************************************
def getLrGH2(w, r, wGrads, dtype=TFFP_TYPE):
    print( '\nRunning SVD to get initial values for G and JH...')
    u, s, vT = np.linalg.svd(w, full_matrices=False)
    ssr = np.sqrt(s[:r])
    g = u[:,:r]*ssr                                     # mxr
    hT = np.diag(ssr).dot(vT[:r,:])                     # rxn

    if dtype in [np.float32, tf.float32]:
        npType = np.float32
        tfType = tf.float32
    else:
        npType = np.float64
        tfType = tf.float64

    n,m = w.shape
    batchSize = 784
    numEpochs = 2000     # Set to zero to bypass the optimization below and use just the results of SVD
#    numEpochs = 0

#    learningRateInit = .01
#    learningRateMin = .0000001
    learningRateInit = .01
    learningRateMin = .0001
    LEARNING_DECAY_RATE = .99
    learningDecayStep = (((n//batchSize)*numEpochs)//
                         (np.log(learningRateMin/learningRateInit)/np.log(LEARNING_DECAY_RATE)))

    SEED = 1234
    lrGraph = tf.Graph()
    with lrGraph.as_default():
        tfIdIndexes = tf.placeholder( tf.int32, shape=[None,1], name='IdIndexes' )
        tfIdColumns = tf.reshape(tf.one_hot(tfIdIndexes, n, dtype=tfType), [-1,n])
        tfWColumns = tf.placeholder( tfType, shape=[None,m], name='WColumns' )
        tfWGrads = tf.placeholder( tfType, shape=[None,m], name='WGrads' )
        
        tfBatch = tf.Variable(0,name='BatchCounter')   # Incremented once per batch
        tfLearningRate = tf.train.exponential_decay(learningRateInit,       # Base learning rate.
                                                    tfBatch,                # Current index into the dataset.
                                                    learningDecayStep,      # Decay step.
                                                    LEARNING_DECAY_RATE,    # Decay rate.
                                                    staircase=True)
                                 
        tfG = tf.Variable(g, dtype=tfType, name='G')
        tfHT = tf.Variable(hT, dtype=tfType, name='HT')
#        tfG = tf.Variable(tf.truncated_normal([n, r], mean=0, stddev=1.0/np.sqrt(n), dtype=tfType, seed=SEED), name='G')
#        tfHT = tf.Variable(tf.truncated_normal([r, m], mean=0, stddev=1.0/np.sqrt(r), dtype=tfType, seed=SEED), name='HT')

        tfWdotIdColumns = tf.matmul( tf.matmul(tfIdColumns, tfG), tfHT)
        tfLoss = tf.reduce_mean(tf.square(tfWColumns - tfWdotIdColumns)*tfWGrads)

        tfOpt = tf.train.AdamOptimizer(learningRateInit).minimize(tfLoss, global_step=tfBatch)

    tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=tfConfig, graph=lrGraph)
    session.run( [v.initializer for v in lrGraph.get_collection( tf.GraphKeys.GLOBAL_VARIABLES)] )

    allIdx = np.arange(n)
    # Calculating initial error:
    a = 0
    sumLoss = 0.0
    while a < n:
        idx = allIdx[a:a+batchSize]
        sumLoss += session.run(tfLoss,
                               feed_dict={
                                            tfIdIndexes:    idx.reshape([batchSize,1]),
                                            tfWColumns:     (npType)(w[idx]),
                                            tfWGrads:       (npType)(wGrads[idx])
                                         })
        a += batchSize
    avgLoss = sumLoss*batchSize/n
    print('MSE After SVD: %.9f'%(avgLoss))

    if numEpochs==0:
        # No Gradient Descent fine-tuning:
        session.close()
        return g,hT,r,avgLoss

    print('Now optimizing for best G, JH...')
    print('r=%d, Num Epoch: %d, BatchSize: %d'%(r,numEpochs,batchSize))
    
    minLoss = 0
    # Now Start Training:
    for ep in range(numEpochs):
#        np.random.shuffle(allIdx)
        a = 0
        sumLoss = 0.0
        while a < n:
            idx = allIdx[a:a+batchSize]
            wColumns = w[idx]
            _,loss = session.run([tfOpt, tfLoss],
                                 feed_dict={
                                            tfIdIndexes:    idx.reshape([batchSize,1]),
                                            tfWColumns:     (npType)(wColumns),
                                            tfWGrads:       (npType)(wGrads[idx])
                                 })
            sumLoss += loss
            a += batchSize
        avgLoss = sumLoss*batchSize/n
        print('  Epoch %d: Avg Loss: %.9f'%(ep+1, avgLoss))
        if minLoss==0:
            minLoss = avgLoss
            g,hT = session.run([tfG, tfHT])
        elif avgLoss<minLoss:
            minLoss = avgLoss
            g,hT = session.run([tfG, tfHT])

    print('Final MSE: %.9f'%(avgLoss))
    print('Min MSE: %.9f'%(minLoss))

    session.close()
    return g,hT,r,minLoss

# **********************************************************************************************************************
def reconW(g, jh, e, f):
    r,m = g.shape
    n = jh.shape[1]
    
    reconGraph = tf.Graph()
    with reconGraph.as_default():
        tfIdIndexes = tf.placeholder( tf.int32, shape=[None,1], name='IdIndexes' )
        tfIdColumns = tf.reshape(tf.one_hot(tfIdIndexes, n, dtype=TFFP_TYPE), [-1,n])
        tfG = tf.constant(g, TFFP_TYPE)
        tfJH = tf.constant(jh, TFFP_TYPE)
        tfReconW = getLdrGraph(False, tfIdColumns, tfG, tfJH, e, f, m , n, r, TFFP_TYPE)

    tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=tfConfig, graph=reconGraph)

    reconW = np.zeros((n,m), dtype=np.float64)
    batchSize = 256
    allIdx = np.arange(n)
    for a in range(0,n,batchSize):
        idx = allIdx[a:a+batchSize].reshape((-1,1))
        reconW[a:a+batchSize,:] = session.run(tfReconW, feed_dict={ tfIdIndexes: idx })
    session.close()
    return reconW

# **********************************************************************************************************************
def getLdrGHOld(w, e, f, r):
#    # Returning random G anf jH:
#    np.random.seed(42) # make results repeatable
#    g = np.float32(np.random.randn(r,m))
#    jh = np.float32(np.random.randn(r,n))
#    return g,jh,e,f,r,0
    
    # Obtaining G and H using SVD on the Stein Displacement matrix of W
    w = np.transpose(w)
    m,n = w.shape
    
    lm = getSteingDisp(w, e, f)
    u, s, vT = np.linalg.svd(lm, full_matrices=False)
    for ss in s: print(ss)
    g = np.transpose(u[:,:r])                           # (r,m)
    jh = np.flip( np.diag(s[:r]).dot(vT[:r,:]),-1)      # (r,n)
    i = np.identity(n)

    return g,jh,e,f,r,0

#    myGraph = tf.Graph()
#    with myGraph.as_default():
#        input = tf.placeholder( TFFP_TYPE, shape=[None,n], name='input' )
#        tfG = tf.constant(g, TFFP_TYPE)
#        tfJH = tf.constant(jh, TFFP_TYPE)
#        try:
#            output = getLdrGraph(False, input, tfG, tfJH, e, f, m, n, r)
#        except NotImplementedError as err:
#            print('\n\nFatal Error: e,f=%d,%d Not Supported for %dx%d Matrix!'%(e,f,m,n))
#            exit(0)
#
#        tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
#                                  intra_op_parallelism_threads=8,
#                                  gpu_options=tf.GPUOptions(allow_growth=True))
#        session = tf.Session(config=tfConfig, graph=myGraph)
#        wI = np.transpose(session.run( output, feed_dict={input:i}))
#        session.close()
#
#        mae = np.abs(wI-w).mean()
#
#    return g,jh,e,f,r,mae

# **********************************************************************************************************************
def getBestLdr(w, rValues):
    if type(rValues) != list:   rValues = [rValues]
    w = np.transpose(w)
    m,n = w.shape
    
    if m<n:
        # (e=0 and f!=0) or m<n
        efSet = [(0, 1), (0,-1), (0,2), (0,-2), (-1,1), (1,-1), (-2,2), (2,2), (2,-2), (-3,3), (3,-3)]
    else:
        # (e!=0 and f=0) or m>n
        efSet = [(1, 0), (-1,0), (2,0), (-2,0), (-1,1), (1,-1), (-2,2), (2,2), (2,-2), (-3,3), (3,-3)]

    i = np.identity(n)
    best = None
    for r in rValues:
        print('Finding best: e,f for %dx%d matrix with rank %d:'%(m,n,r))
        for (e,f) in efSet:
            lm = getSteingDisp(w, e, f)
            u, s, vT = np.linalg.svd(lm, full_matrices=True)
            g = np.transpose(u[:,:r])                           # (r,m)
            jh = np.flip( np.diag(s[:r]).dot(vT[:r,:]),-1)      # (r,n)
            
            myGraph = tf.Graph()
            with myGraph.as_default():
                input = tf.placeholder( TFFP_TYPE, shape=[None,n], name='input' )
                tfG = tf.constant(g, TFFP_TYPE)
                tfJH = tf.constant(jh, TFFP_TYPE)
                try:
                    output = getLdrGraph(False, input, tfG, tfJH, e, f, m, n, r)
                except NotImplementedError as err:
                    print('    Trying e,f=%d,%d => Not Supported!'%(e,f))
                    continue

            tfConfig = tf.ConfigProto(inter_op_parallelism_threads=8,
                                      intra_op_parallelism_threads=8,
                                      gpu_options=tf.GPUOptions(allow_growth=True))
            session = tf.Session(config=tfConfig, graph=myGraph)
            wI = np.transpose(session.run( output, feed_dict={input:i}))
            session.close()

            err = np.abs(wI-w).mean()
            if best is None:
                best = (e, f, g, jh, r, err)
            elif best[5]>err:
                best = (e, f, g, jh, r, err)

            print('    Trying e,f=%d,%d => MAE: %f'%(e,f,err))

    print('Best: e,f=%d,%d, r=%d, MAE= %f\n'%(best[0], best[1], best[4], best[5]))
    return best[0], best[1], best[2], best[3], best[4]

#m,n=16,8
#w = np.float32(np.reshape(range(m*n), (n,m)))+1
#getBestLdr(w, 3)
#exit(0)

# **********************************************************************************************************************
if __name__ == '__main__':
    print('\n\nTesting Inference Mode:')
    testLdrGraph(False)
#    print('\n\nTesting Training Mode:')
#    testLdrGraph(True)
    exit(0)

#    print('\n\nProfiling Inference Mode:')
#    profileLdrGraph(False)
#    print('\n\nProfiling Training Mode:')
#    profileLdrGraph(True)
#    exit(0)

