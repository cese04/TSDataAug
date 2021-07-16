import numpy as np


# Usado para ruido de EOG
ruido_eog = np.load("ruido.npy")
#ruidoEOG = np.load("data/ruidoEOG_750.npy")

# Metodo para formar ventanas dado una longitud y un traslape


def ventaneo(senal, l=20, t=5):
  # l: largo de la ventana
  # t: traslape
    ventanas = []
    mx = np.shape(senal)[1]
    beg = 0
    ds = l
    # print(mx)
    i = 0
    # Mientras no llegue al limite, saca más ventanas
    while ds < mx:
        #print(beg, "----", ds)
        ventanas.append(senal[:, (l-t)*i: (l-t)*i + l])
        i += 1
        beg += (l-t)*i
        ds = (l-t)*i + l

    return np.array(ventanas)


def getSamples(X_train, y_train, samples=10, ruido="gauss", ns=100):
    # Tomar indices aleatorios de la señal
    smp = np.random.randint(0, len(X_train), samples)
    X_train_v = []
    y_train_v = []
    # De cada muestra aleatoria sacar ventanas
    for s in smp:
        # Elegir una muestra aleatoriamente
        ventana = ventaneo(X_train[s],
                            l=250*4,
                            t=250*2+450)
        X_train_v.append(ventana)
        y_train_v.append(np.tile(y_train[s], len(ventana)))
    X_train_v = np.array(X_train_v)
    shp = np.shape(X_train_v)
    X_train_v = X_train_v.reshape((shp[0] * shp[1], shp[2], shp[3]))
    

    # Sumar el ruido acorde al caso
    if ruido == "gauss":
        ruido = np.random.randn(*X_train_v.shape)
        X_train_v += ruido * 2

    if ruido == "eog":
        for i in range(len(X_train_v)):
            ri, rj = np.random.randint(162-22), np.random.randint(96735-1000)
            X_train_v[i] += ruido_eog[ri:ri + 22, rj:rj + 1000] * ns
        #print("eog")

    return X_train_v, y_train_v

def getSamplOrig(X_train, samples=10, ruido='gauss', ns=100):
    """ Devuelve la senal con ruido y la original """

    # Indices aleatorips
    smp = np.random.randint(0, len(X_train), samples)
    # Señales con ruido
    #X_train_r = []
    # Señales originales
    X_train_o = []

    for s in smp:
        # Elegir una muestra aleatoriamente
        ventana = ventaneo(X_train[s],
                            l=250*4,
                            t=250*2+450)
        X_train_o.append(ventana)

    X_train_o = np.array(X_train_o)
    shp = np.shape(X_train_o)
    X_train_o = X_train_o.reshape((shp[0] * shp[1], shp[2], shp[3]))

    # Señal con ruido
    X_train_r = X_train_o.copy()

    # Sumar el ruido acorde al caso
    if ruido == "gauss":
        ruido = np.random.randn(*X_train_r.shape)
        X_train_r += ruido * 2

    if ruido == "eog":
        for i in range(len(X_train_r)):
            ri, rj = np.random.randint(162-22), np.random.randint(96735-1000)
            X_train_r[i] += ruido_eog[ri:ri + 22, rj:rj + 1000] * ns
        #print("eog")

    return X_train_r, X_train_o


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from moabb.datasets import BNCI2014001
    from moabb.paradigms import (LeftRightImagery, MotorImagery,
                             FilterBankMotorImagery)
    # Extraer los datos
    dataset = BNCI2014001()
    # Cargar el paradigma a usar
    paradigm = MotorImagery(n_classes=4,
                        fmin=8,
                        fmax=35,
                        tmin=-2,
                        tmax=5)
                        
    X, y, metadata = paradigm.get_data(
    dataset=dataset, subjects=[1])
    mxx = np.max(X) / 5
    X = (X / mxx)

    #samples, _ = getSamples(X, y, 10, "eog", 11000)
    samples, originals = getSamplOrig(X, 2, "eog", 5100)

    print(np.shape(samples))
    #plt.subplot(121)
    plt.plot(samples[0, 0])
    #plt.subplot(122)
    plt.plot(originals[0, 0])
    plt.show()
