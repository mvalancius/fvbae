
import keras
import numpy as np

def vae(Xtrain, nndim = 200, latentdim = int(2), samplestd = 1.0,
        epochs = int(10), batch_size = int(100), optimizer = "Adamax", return_type = "all"):
    """
    vae(Xtrain, nndim = 200, latentdim = int(2), samplestd = 1.0, epochs = int(10), batch_size = int(100), optimizer = "Adamax", return_type = "all")
    
    Add arguments element wise.
    
    Parameters
    -----------
    Xtrain : 2D numpy array
        Rows are observations and columns are features. Only float values.
    nndim : integer
        number of hidden units in neural network. 
    latentdim : integer
        Number of latent dimensions. Must be less than Xtrain feature size and nndim.
    samplestd : float
        standard deviation of normal distribution chosen for latent dimension, where mean is zero.
    epochs : integer
        Passes through dataset, as defined by Keras. Must be greater than 0
    batch_size : integer
        Mini-batch size. Must be greater than zero and less than or equal to number of observations.
    optimizer : string of Keras optimizer or Keras optimizer object
        Optimizer passed to Keras. If string, defaults for that optimizer are chosen.
        If using Keras optimizer object, more control (such as learning rate) is allowed.
        Follows typical Keras notation. Further details available here: https://keras.io/optimizers/
    return_type : string
        Either "autoencoder", "encoder", "decoder" or other string / None.
        Allows for the return of only one of the Keras objects. If none of
        "autoencoder", "encoder", or "decoder" is specified, all are returned.
        
    Returns
    -----------
    Trained Keras model. Return type specified by return_type parameter. 
        Model can then be used as if it were a manually trained Keras model and all usual methods apply
        
    Notes
    --------
    When using predict method on returned model, ensure batch size is the same as specified in vae function
    
    Example:
    
    >>> X = np.random.normal(loc = 1, scale = 2, size = (100000, 500))
    >>> opt = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
    autoencoder, encoder, decoder = fvbae.vae(X, nndim = 100, latentdim = 5, optimizer = opt,
        epochs = 100, batch_size = int(100))
        
    """
    
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
