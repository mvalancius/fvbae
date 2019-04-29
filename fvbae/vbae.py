
import keras
import numpy as np

def vae(Xtrain, nndim = 200, latentdim = int(2), samplestd = 1.0,
        epochs = int(10), batch_size = int(100), optimizer = "Adamax", return_type = "all"):
    """Implementation of an auto-encoder following AEVB"""
    
    inputdim = Xtrain.shape[1]

    ### Converting integer parameters to integers
    epochs = int(epochs)
    batch_size = int(batch_size)
    
       
    ### Checking for correct dimensions    
    if inputdim <= nndim or inputdim <= latentdim:
        raise Exception("Overspecified model")
    
    if Xtrain.shape[0] < 2:
        raise Exception("Need multiple training observations to fit model")
    
    if batch_size > Xtrain.shape[0] or batch_size < 1:
        raise Exception("Batch size must be less than number of observations")
    
    if epochs < 1:
        raise Exception("Must have at least one epoch")


    X = keras.layers.Input(shape = (inputdim,))
    Q = keras.layers.Dense(units = nndim, activation = "relu")(X)
    Zm = keras.layers.Dense(units = latentdim)(Q)
    Zv = keras.layers.Dense(units = latentdim)(Q)

    def lam(det):
        """Reparameterization trick"""
        Zm, Zv = det
        noise = keras.backend.random_normal(shape = (batch_size, latentdim),
                                       mean = 0.0,
                                       stddev=samplestd)
        return Zm + noise * keras.backend.sqrt(keras.backend.exp(Zv))

    Z = keras.layers.Lambda(lam, output_shape = (latentdim,))([Zm, Zv])

    Qinv = keras.layers.Dense(units = nndim, activation = "relu")
    Xinv = keras.layers.Dense(units = inputdim, activation = "sigmoid")
    Qout = Qinv(Z)
    Xout = Xinv(Qout)


    ### Generating from latent space
    Zgen = keras.layers.Input(shape = (latentdim,))
    Qgen = Qinv(Zgen)
    Xgen = Xinv(Qgen)

    ### Defining the 3 models
    autoencoder = keras.Model(X, Xout)
    encoder = keras.Model(X, Zm)
    decoder = keras.Model(Zgen, Xgen)


    l= 1.0

    ### Creation of loss function
    def AEVBloss(X, Xout):
        """Loss from Auto-Encoding Variational Bayes page 5"""
        decterm = (inputdim/l) * keras.losses.binary_crossentropy(X, Xout) 
        KLterm = -0.5 * keras.backend.mean(1 + Zv - 
                                       keras.backend.square(Zm) - 
                                      keras.backend.exp(Zv),
                                      axis = -1)
        return KLterm + decterm


    autoencoder.compile(optimizer = optimizer, loss = AEVBloss)

    autoencoder.fit(x = Xtrain, y = Xtrain, epochs = epochs, batch_size = batch_size)
    
    ### Allowing for different return objects
    if return_type == "autoencoder":
        return autoencoder
    if return_type == "encoder":
        return encoder
    if return_type == "decoder":
        return decoder

    return([autoencoder, encoder, decoder])
