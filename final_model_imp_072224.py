import os
import warnings
import time

# Specific
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import gpflow
from gpflow.utilities import print_summary, set_trainable, deepcopy
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from matplotlib import pyplot as plt

# =============================================================================
# Auxiliary Functions
# =============================================================================


def set_white_exp_95CI(code):
    """
    Sets the estimated average ~95% confidence interval on labels using the property code

    Parameters:
    code : string
        Property code

    Returns:
    exp_95CI : float
    """
    if code == 'Tb':
        exp_95CI = 10.0
    elif code == 'Tm':
        exp_95CI = 5.0
    elif code == 'Hvap':
        exp_95CI = 1.0
    elif code == 'Vc':
        exp_95CI = 25.0
    elif code == 'Tc':
        exp_95CI = 5.0
    elif code == 'Pc':
        exp_95CI = 30.0
    return exp_95CI



def gpConfig_from_method(method_number, code, kernel = 'RBF', anisotropic = False, useWhiteKernel = True, trainLikelihood = True, opt_method = 'L-BFGS-B'):
    """
    Creates a gpConfig dictionary based on the method number.

    Parameters:
    method_number : int
        Method number.

    Returns:
    gpConfig : dictionary
        Dictionary of GP configuration parameters.

    Note:
    method_number is used to define which type of gp model to use
    1: Y = GP(0, K(Mw, Y_gc))
    2: Y - Y_gc = GP(0, K(Mw))
    3: Y - Y_gc = GP(0, K(Mw, Y_gc))
    4: Y = GP(Y_gc, K(Mw, Y_gc))
    5: Y = GP(AMw + BY_gc + c, K(Mw, Y_gc))
    """
    gpConfig={'kernel': kernel,
           'useWhiteKernel':useWhiteKernel,
           'trainLikelihood':trainLikelihood,
           'opt_method':opt_method,
           'anisotropic':anisotropic}
    if method_number == 1:
        gpConfig['mean_function']='Zero'
        gpConfig['Name']='y_exp = GP(0, K(x1,x2))'
        gpConfig['SaveName']='model_1'
    if method_number == 2:
        gpConfig['mean_function']='Zero'
        gpConfig['Name']='y_exp = y_GC + GP(0, K(x1))'
        gpConfig['SaveName']='model_2'
    if method_number == 3:
        gpConfig['mean_function']='Zero'
        gpConfig['Name']='y_exp = y_GC + GP(0, K(x1,x2))'
        gpConfig['SaveName']='model_3'
    if method_number == 4:
        gpConfig['mean_function']='Constant'
        gpConfig['Name']='y_exp = GP(y_GC, K(x1,x2))'
        gpConfig['SaveName']='model_4'
    if method_number == 5:
        gpConfig['mean_function']='Linear'
        gpConfig['Name']='y_exp = GP(B@X, K(x1,x2))'
        gpConfig['SaveName']='model_5'
    else: 
        if method_number not in [1 , 2, 3, 4, 5]:
            raise ValueError('invalid method number input')   
    return gpConfig


def get_gp_data(X, Y, method_number):
    """
    Gets X and Y data to train GP based on the method number

    Parameters:
    X : numpy array
        Features data.
    Y : numpy array
        Property data.
    method_number : int
        Method number

    Returns:
    X_gp : numpy array
        Features data to train GP.
    Y_gp: numpy array
        Data to train GP.
    Y_gc: numpy array
        Data from Joback method
    """
    if method_number == 2:
        X_gp = X[:,0].reshape(-1,1)
    else:
        X_gp = X
    if method_number in [2,3]:
        Y_gp = Y.flatten() - X[:,1]
    else:
        Y_gp = Y
    Y_gp = Y_gp.reshape(-1,1)
    Y_gc = X[:,1].reshape(-1,1)
    return X_gp, Y_gp, Y_gc

def discrepancy_to_property(method_number, y_pred, y_gc, idx):
    """
    Adds discrepancy to property based on the method number
    Parameters:
    method_number : int
        Method number
    y_pred : numpy array
        GP predicted output
    y_gc : numpy array
        Predicted GC method results
    idx : np.array
        Index of the y_gc to be added to y_pred

    Returns:
    y_prop : numpy array
        Predicted property value
    """
    if method_number in [2,3]:
        y_prop = y_pred + y_gc[idx.flatten(),:]
    else:
        y_prop = y_pred
    return y_prop

def stratifyvector(Y):
    """
    Creates a stratified vector based on the label data Y

    Parameters:
    Y : numpy array
        label data
    Returns:
    stratifyVector : numpy array
        Stratified vector
    """
    # Iterate over number of bins, trying to find the larger number of bins that
    # guarantees at least 5 values per bin
    for n in range(1,100):
        # Bin Y using n bins
        stratifyVector=pd.cut(Y,n,labels=False)
        # Define isValid (all bins have at least 5 values)
        isValid=True
        # Check that all bins have at least 5 values
        for k in range(n):
            if np.count_nonzero(stratifyVector==k)<5:
                isValid=False
        #If isValid is false, n is too large; nBins must be the previous iteration
        if not isValid:
            nBins=n-1
            break
    # Generate vector for stratified splitting based on labels
    stratifyVector=pd.cut(Y,nBins,labels=False)
    return stratifyVector

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : numpy array
        Array to be normalized. If dim>1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (0,1)
            . LogStand - standardization on the log of the variable,
                         (log(x)-mean(log(x)))/std(log(x))
            . Log+bStand - standardization on the log of variables that can be
                           zero; uses a small buffer,
                           (log(x+b)-mean(log(x+b)))/std(log(x+b))
        Default: 'Standardization'
    reverse : bool
        Whether  to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=np.log(inputArray)
        elif method=='Log+bStand': aux=np.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        if method!='MinMax':
            skScaler=preprocessing.StandardScaler().fit(aux)
        else:
            skScaler=preprocessing.MinMaxScaler().fit(aux)
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand': inputArray=np.exp(inputArray)
        elif method=='Log+bStand': inputArray=np.exp(inputArray)-10**-3
    elif not reverse:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=np.log(inputArray)
        elif method=='Log+bStand': aux=np.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        inputArray=skScaler.transform(aux)
    # Return
    return inputArray,skScaler


tf.config.run_functions_eagerly(True)
def bounded_parameter(low, high, initial_value):
    sigmoid = tfb.Sigmoid(low=tf.cast(low, dtype=tf.float64), high=tf.cast(high, dtype=tf.float64))
    return gpflow.Parameter(initial_value, transform=sigmoid, dtype=tf.float64)


def buildGP(X_Train, Y_Train, gpConfig, code, sc_y_scale, featurenorm, retrain_count):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    Y_Train : numpy array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern12' - gpflow.kernels.Matern12()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.
    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RQ')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    typeMeanFunc=gpConfig.get('mean_function','Zero')
    opt_method=gpConfig.get('opt_method','L-BFGS-B')
    anisotropy=gpConfig.get('anisotropic','False')
    
    
    if retrain_count == 0:
        lengthscale_1 = bounded_parameter(0.00001, 100.0, 1.0)
        if anisotropy == True:
            lengthscale_ = np.ones(X_Train.shape[1])*lengthscale_1
        else:
            lengthscale_ = lengthscale_1
        variance_ = bounded_parameter(0.00001, 100.0, 1.0)
        alpha_ = bounded_parameter(0.0001, 3000, 1.0)
        white_var = 1.0
    else:
        seed_ = int(retrain_count) * 100
        np.random.seed(seed_)
        tf.random.set_seed(seed_)
        lengthscale_1 = bounded_parameter(0.00001, 100.0, np.array(np.random.uniform(0.1, 100.0), dtype='float64'))
        if anisotropy == True:
            lengthscale_ = np.ones(X_Train.shape[1])*lengthscale_1
        else:
            lengthscale_ = lengthscale_1
        variance_ = bounded_parameter(0.00001, 100.0, np.array(np.random.lognormal(0.0, 1.0), dtype='float64'))
        alpha_ = bounded_parameter(0.0001, 3000, np.array(np.random.uniform(0.01, 100), dtype='float64'))
        white_var = bounded_parameter(0.00001, 10, np.array(np.random.uniform(0.05, 10), dtype='float64'))
        #white_var = 1.0
    
#         if featurenorm == "Standardization":
#             exp_95ci = set_white_exp_95CI(code)
#             white_var = ((exp_95ci/1.96)**2)/(sc_y_scale**2)
#             white_var = float(white_var)
        
    # Select and initialize kernel
    if kernel=='RBF':
        gpKernel=gpflow.kernels.SquaredExponential(variance=variance_, lengthscales=lengthscale_)
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic(variance=variance_, lengthscales=lengthscale_, alpha=alpha_)
    if kernel=='Matern12':
        gpKernel=gpflow.kernels.Matern12(variance=variance_, lengthscales=lengthscale_)
    if kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32(variance=variance_, lengthscales=lengthscale_)
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52(variance=variance_, lengthscales=lengthscale_)
    # Add White kernel
    if useWhiteKernel: 
        if featurenorm == "Standardization":
            gpKernel=gpKernel+gpflow.kernels.White(variance = white_var)
        else:
            gpKernel=gpKernel+gpflow.kernels.White()
            
    # Add Mean function
    if typeMeanFunc == 'Zero':
        mf = None
    if typeMeanFunc == 'Constant':
        #If constant value is selected but no value is given, default to zero mean
        mf_val = np.array([0,1]).reshape(-1,1)
        mf = gpflow.functions.Linear(mf_val)
    if typeMeanFunc == 'Linear':
        A = np.ones((X_Train.shape[1],1))
        mf = gpflow.functions.Linear(A)
    # Build GP model    
    model=gpflow.models.GPR((X_Train,Y_Train),gpKernel,mean_function=mf, noise_variance=10**-5)
    model_pretrain = deepcopy(model)
    # print(gpflow.utilities.print_summary(model))
    condition_number = np.linalg.cond(model.kernel(X_Train))
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    #Set A to be non-trainable for method 4
    if typeMeanFunc == 'Constant':
        gpflow.set_trainable(model.mean_function.A, False)
        gpflow.set_trainable(model.mean_function.b, False)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()
    # Fit GP to training data

    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                           options={'maxiter':10**9},
                           method=opt_method)
    obj_func = model.training_loss()
    if aux.success:
        opt_success = True
    else:
        opt_success = False
    # Check convergence
#     if aux.success==False:
#         warnings.warn('GP optimizer failed to converge.')

    # print(gpflow.utilities.print_summary(model))
    return model, aux, condition_number, obj_func, opt_success, retrain_count, model_pretrain



def train_gp(X_Train, Y_Train, gpConfig, code, sc_y_scale, featurenorm, retrain_GP, retrain_count):
    """
    Trains the GP given training data. Sets self.trained_hyperparams and self.fit_gp_model
    
    Notes:
    ------
    Sets the following parameters of self
    self.trained_hyperparams: list, the trained hyperparameters of the GP model
    self.fit_gp_model: instance of gpflow.models.GPR, the trained GP model
    self.posterior: instance of gpflow.mean_field.KFGaussian, the posterior of the GP model 
    """ 

    # Train the model multiple times and keep track of the model with the lowest minimum training loss
    best_minimum_loss = float('inf')
    best_model = None
    best_model_pretrain = None
    best_model_success = False
    best_condition_num = float('inf')
    args = (X_Train, Y_Train, gpConfig)
    
    retrain_GP = int(retrain_GP)
    retrain_count = retrain_count
    for i in range(retrain_GP):
        model, aux, condition_number, obj_func, opt_success, retrain_count, model_pretrain = \
            buildGP(X_Train, Y_Train, gpConfig, code, sc_y_scale, featurenorm, retrain_count)
        print(f"training_loss = {obj_func}")
        print(f"condition_number = {condition_number}")
        retrain_count += 1
        if best_minimum_loss > obj_func and opt_success==True:
            best_minimum_loss = obj_func
            best_model = model
            best_model_pretrain = model_pretrain
            best_model_success = opt_success
            best_condition_num = condition_number
    if best_model_success == False:
        warnings.warn('GP optimizer failed to converge with retrains')
    
    #Put hyperparameters in a list
    trained_hyperparams = gpflow.utilities.read_values(best_model)

    return best_model,best_minimum_loss,best_model_success,best_condition_num,trained_hyperparams,best_model_pretrain,sc_y_scale


def gpPredict(model,X):
    """
    gpPredict() returns the prediction and standard deviation of the GP model
    on the X data provided.

    Parameters
    ----------
    model : gpflow.models.gpr.GPR object
        GP model.
    X : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).

    Returns
    -------
    Y : numpy array (N,1)
        GP predictions.
    STD : numpy array (N,1)
        GP standard deviations.

    """
    # Do GP prediction, obtaining mean and variance
    GP_Mean,GP_Var=model.predict_f(X)
    # Convert to numpy
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Prepare outputs
    Y=GP_Mean
    STD=np.sqrt(GP_Var)
    # Output
    return Y,STD

# =============================================================================
# Plots
# =============================================================================
def save_fig(save_path, ext='png', close=True):                
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(save_path)[0]
    filename = "%s.%s" % (os.path.split(save_path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    # Actually save the figure
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    
    # Close it
    if close:
        plt.close()

        
        
def parity_plot(code, Y_Train, Y_Test, Y_Train_Pred, Y_Test_Pred, Y_Train_CI, Y_Test_CI, gpConfigs, save_plot = False, disc = False):
    """
    parity_plot() generates a parity plot comparing the experimental and predicted property values

    Parameters:
    code : string
        Property code
    Y_Train : numpy array
        Training set property values
    Y_Test : numpy array
        Testing set property values
    Y_Train_Pred : numpy array
        Training set predicted property values
    Y_Test_Pred : numpy array   
        Testing set predicted property values
    gpConfigs : dictionary   
        Dictionary of GP Configuration
    save_plot : bool
        Whether to save the plot or not
    disc : bool
        Whether to plot the discrepancy of the property or not
    """
    # Pyplot Configuration
    plt.rcParams['figure.dpi']=300
    plt.rcParams['savefig.dpi']=300
    plt.rcParams['text.usetex']=False
    plt.rcParams['font.family']='serif'
    # plt.rcParams['font.serif']='Times New Roman'
    plt.rcParams['font.weight']='bold'
    plt.rcParams['mathtext.rm']='serif'
    plt.rcParams['mathtext.it']='serif:italic'
    plt.rcParams['mathtext.bf']='serif:bold'
    plt.rcParams['mathtext.fontset']='custom'
    plt.rcParams['axes.titlesize']=9
    plt.rcParams['axes.labelsize']=9
    plt.rcParams['xtick.labelsize']=9
    plt.rcParams['ytick.labelsize']=9
    plt.rcParams['font.size']=6
    plt.rcParams["savefig.pad_inches"]=0.02

    # Get target variable denominator
    if code=='Tb':
        varName='T$_{b}$ /K'
    elif code=='Tm':
        varName='T$_{melt}$ /K'
    elif code=='Hvap':
        varName = "H$_{vap}$ /kJmol$^{-1}$"
    elif code == "Vc":
        varName = 'V$_{c}$ /cm$^3$mol$^{-1}$'
    elif code == "Tc":
        varName = 'T$_{c}$ /K'
    elif code == "Pc":
        varName = 'P$_{c}$ /bar'
    else:
        varName = 'Property'
    # Predictions Scatter Plot
    # Compute metrics
    try:
        R2_Train=metrics.r2_score(Y_Train,Y_Train_Pred)
        R2_Test=metrics.r2_score(Y_Test,Y_Test_Pred)
    except:
        R2_Train = None
        R2_Test = None
    try:
        MAE_Train=metrics.mean_absolute_error(Y_Train,Y_Train_Pred)
        MAE_Test=metrics.mean_absolute_error(Y_Test,Y_Test_Pred)
    except:
        MAE_Train = None
        MAE_Test = None
    
    MAPD_Train = None
    MAPD_Test = None
    if disc != True:
        try:
            MAPD_Train=metrics.mean_absolute_percentage_error(Y_Train,Y_Train_Pred)*100
            MAPD_Test=metrics.mean_absolute_percentage_error(Y_Test,Y_Test_Pred)*100
        except:
            pass

    # Plot
    plt.figure(figsize=(2.3,2))
    plt.plot(Y_Train,Y_Train_Pred,'ow',markersize=3,  mec ='r', mew=0.5, zorder = 2)
    train_err = plt.errorbar(Y_Train.flatten(), Y_Train_Pred.flatten(), yerr = Y_Train_CI.flatten(), 
                 ls='none', color = 'red', linewidth=0.5, zorder = 1, alpha = 0.5) 
    (plotline, _, _) = train_err
    train_err[-1][0].set_linestyle('--')
    plotline.set_markerfacecolor('none')
    plt.plot(Y_Test,Y_Test_Pred,'^w',markersize=3, mec ='b', mew= 0.5, zorder = 4)
    test_err = plt.errorbar(Y_Test.flatten(), Y_Test_Pred.flatten(), yerr = Y_Test_CI.flatten(), 
                 ls='none', color = 'b', linewidth=0.5, zorder = 3, alpha = 0.5) 
    (plotline2, _, _) = test_err
    test_err[-1][0].set_linestyle('--')
    plotline2.set_markerfacecolor('none')
    
    lims=[np.min([plt.gca().get_xlim(),plt.gca().get_ylim()]),
        np.max([plt.gca().get_xlim(),plt.gca().get_ylim()])]
    plt.axline((lims[0],lims[0]),(lims[1],lims[1]),color='k',
            linestyle='--',linewidth=1)

    disc_save = ""
    disc_title = ""
    plt.xlabel('Exp. '+varName + disc_title,weight='bold')
    plt.ylabel('Pred. '+varName + disc_title,weight='bold')
    
    plt.title(gpConfigs["Name"])
    
    if MAE_Train != None:
        plt.text(0.03,0.93,
                'MAE = '+'{:.2f} '.format(MAE_Train)+varName.split()[-1][1:],
                horizontalalignment='left',
                transform=plt.gca().transAxes,c='r')
    if MAE_Test != None:
        plt.text(0.03,0.87,
            'MAE = '+'{:.2f} '.format(MAE_Test)+varName.split()[-1][1:],
            horizontalalignment='left',
            transform=plt.gca().transAxes,c='b')
    if MAPD_Train != None:
        plt.text(0.03,0.81,
                'MAPD = '+'{:.2f} '.format(MAPD_Train)+"%",
                horizontalalignment='left',
                transform=plt.gca().transAxes,c='r')
    if MAPD_Test != None:
        plt.text(0.03,0.75,
            'MAPD = '+'{:.2f} '.format(MAPD_Test)+"%",
            horizontalalignment='left',
            transform=plt.gca().transAxes,c='b') 
    if R2_Train != None:
        plt.text(0.97,0.09,'$R^2$ = '+'{:.2f}'.format(R2_Train),
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='r')
    if R2_Test != None:
        plt.text(0.97,0.03,'$R^2$ = '+'{:.2f}'.format(R2_Test),
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='b')
    if save_plot == True:
        save_path = "Final_Results/" + code + "/" + gpConfigs['SaveName'] +"/" + disc_save + "Parity_Plot"
        save_fig(save_path)
    else:
        plt.show()
                    
    
def parity_plot_final(code, X_data, test_ind, train_ind,
                      Y_Train, Y_Test, Y_Train_Pred, Y_Test_Pred, 
                      gpConfigs, save_plot = False, disc = False):
    """
    parity_plot() generates a parity plot comparing the experimental and predicted property values

    Parameters:
    code : string
        Property code
    Y_Train : numpy array
        Training set property values
    Y_Test : numpy array
        Testing set property values
    Y_Train_Pred : numpy array
        Training set predicted property values
    Y_Test_Pred : numpy array   
        Testing set predicted property values
    gpConfigs : dictionary   
        Dictionary of GP Configuration
    save_plot : bool
        Whether to save the plot or not
    disc : bool
        Whether to plot the discrepancy of the property or not
    """
    # Pyplot Configuration
    plot_dpi = 300
    plt.rcParams['figure.dpi']=plot_dpi
    plt.rcParams['savefig.dpi']=plot_dpi
    plt.rcParams['text.usetex']=False
    plt.rcParams['font.family']='serif'
    # plt.rcParams['font.serif']='Times New Roman'
    plt.rcParams['font.weight']='bold'
    plt.rcParams['mathtext.rm']='serif'
    plt.rcParams['mathtext.it']='serif:italic'
    plt.rcParams['mathtext.bf']='serif:bold'
    plt.rcParams['mathtext.fontset']='custom'
    plt.rcParams['axes.titlesize']=16
    plt.rcParams['axes.labelsize']=16
    plt.rcParams['xtick.labelsize']=16
    plt.rcParams['ytick.labelsize']=16
    plt.rcParams['font.size']=16
    plt.rcParams["savefig.pad_inches"]=0.02
    
    Y_JR_test = X[test_ind,-1].reshape(-1,1)
    Y_JR_train = X[train_ind,-1].reshape(-1,1)

    # Get target variable denominator
    if code=='Tb':
        varName='T$_{b}$ /K'
    elif code=='Tm':
        varName='T$_{melt}$ /K'
    elif code=='Hvap':
        varName = "H$_{vap}$ /kJmol$^{-1}$"
    elif code == "Vc":
        varName = 'V$_{c}$ /cm$^3$mol$^{-1}$'
    elif code == "Tc":
        varName = 'T$_{c}$ /K'
    elif code == "Pc":
        varName = 'P$_{c}$ /bar'
    else:
        varName = 'Property'
    try:
        R2_Train=metrics.r2_score(Y_Train, Y_Train_Pred)
        R2_JR_Train=metrics.r2_score(Y_Train, Y_JR_train)
        R2_Test=metrics.r2_score(Y_Test, Y_Test_Pred)
        R2_JR_Test=metrics.r2_score(Y_Test, Y_JR_test)
    except:
        R2_Train = None
        R2_JR_Train = None
        R2_Test = None
        R2_JR_Test = None
    
    plt.scatter(Y_Train_Pred, Y_Train, marker='o', color='red', alpha=0.4)
    plt.scatter(Y_JR_train, Y_Train, marker='+', color='green', alpha=0.6)
    plt.scatter(Y_Test_Pred, Y_Test, marker='^', color='blue', alpha=0.4)
    plt.scatter(Y_JR_test, Y_Test, marker='x', color='black', alpha=0.4)
    
    
    lims=[np.min([plt.gca().get_xlim(),plt.gca().get_ylim()]),
        np.max([plt.gca().get_xlim(),plt.gca().get_ylim()])]
    plt.axline((lims[0],lims[0]),(lims[1],lims[1]),color='k',
            linestyle='--',linewidth=1)
    if disc == True:
        disc_title = " Discrepancy"
        disc_save = "Discrepancy_"
    else:
        disc_save = ""
        disc_title = ""
    plt.xlabel('Pred. '+varName + disc_title,weight='bold')
    plt.ylabel('Exp. '+varName + disc_title,weight='bold')
    
    #plt.title(gpConfigs["Name"])
    
    if R2_Train != None:
        plt.text(0.99,0.23,'GP train $R^2$ = '+'{:.2f}'.format(R2_Train),
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='red', fontsize=13)
    if R2_JR_Train != None:
        plt.text(0.99,0.16,'GC train $R^2$ = '+'{:.2f}'.format(R2_JR_Train),
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='green', fontsize=13)
    if R2_Test != None:
        plt.text(0.99,0.09,'GP test $R^2$ = '+'{:.2f}'.format(R2_Test),
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='blue', fontsize=13)    
    if R2_JR_Test != None:
        plt.text(0.99,0.02,'GC test $R^2$ = '+'{:.2f}'.format(R2_JR_Test),
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='black', fontsize=13)
    if save_plot == True:
        save_path = "Final_Results/" + code + "/" + gpConfigs['SaveName'] +"/" + disc_save + "Parity_Plot_gcgp"
        save_fig(save_path)
    else:
        plt.show()

        
def count_outside_95(Y_Train, Y_Test, Y_Train_Pred, Y_Test_Pred, Y_Train_CI, Y_Test_CI):
    out_95_train = []
    out_95_test = []
    for index, value in enumerate(Y_Train):
        if np.abs(value - Y_Train_Pred[index]) > Y_Train_CI[index]:
            out_95_train.append(index)
    num_out95_train = len(out_95_train)
    frac_out95_train = num_out95_train/len(Y_Train)
    for index, value in enumerate(Y_Test):
        if np.abs(value - Y_Test_Pred[index]) > Y_Test_CI[index]:
            out_95_test.append(index)
    num_out95_test = len(out_95_test)
    frac_out95_test = num_out95_test/len(Y_Test)
    
    return num_out95_train, frac_out95_train, num_out95_test, frac_out95_test
    

# -*- coding: utf-8 -*-
"""
Script to train a GP on physicochemical properties.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
        . normalize()
        . buildGP()
        . gpPredict()
    . Main Script
    . Plots

Last edit: 2024-07-22
Contributors: Dinis Abranches, Montana Carlozo, Barnabas Agbodekhe
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import warnings
import time
# Specific
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import gpflow
from matplotlib import pyplot as plt
#from utils import normalize, gpPredict, buildGP, parity_plot, fit_GP, train_gp, stratifyvector, get_gp_data, init_hyper_parameters, discrepancy_to_property, gpConfig_from_method

# =============================================================================
# Configuration
# =============================================================================
#Model data is found based on method number

dbPath=""
# Property Code
code='Pc' # 'Hvap', 'Vc', 'Pc', 'Tc', 'Tb', 'Tm'

# Define normalization methods
featureNorm='Standardization' # None,Standardization,MinMax
labelNorm='Standardization' # None,Standardization,LogStand
kernel = 'RQ' #Other Options: RQ, RBF, Matern12, Matern32, Matern52
anisotropic = False
opt_method = 'L-BFGS-B' #Other Options: L-BFGS-B, BFGS
useWhiteKernel = True
trainLikelihood = False
save_plot = True
retrain_GP = 10
method_number = 4

seed = 42
np.random.seed(seed)

# GP Configuration
gpConfig= gpConfig_from_method(method_number, code, kernel, anisotropic, useWhiteKernel, trainLikelihood, opt_method)

# =============================================================================
# Main Script
# =============================================================================

# Iniate timer
ti=time.time()

# Load data
db=pd.read_csv(os.path.join(dbPath,code+'_prediction_data.csv'))
#db=pd.read_csv(os.path.join(dbPath,code+'_prediction_data_correctedOutliers.csv'))
db=db.dropna()
X=db.iloc[:,2:-1].copy().to_numpy('float')
data_names=db.columns.tolist()[2:]
Y=db.iloc[:,-1].copy().to_numpy('float')
Y = Y.reshape(-1,1)
Y_gc = X[:,-1].reshape(-1,1)

#Get X and Y data based on method number
# X_new, Y_new, Y_gc_new = get_gp_data(X, Y, method_number)

#X_stratify = X[:,-1].reshape(-1,1)
X_stratify = X[:,0:]
indices = np.arange(X.shape[0])
Y_stratify = np.column_stack((indices, Y))
X_Train_0, Y_Train_0, X_Test_0, Y_Test_0 = \
                 iterative_train_test_split(X_stratify, Y_stratify, test_size = 0.2)


# Find the indices for the train and test sets
train_indices = (Y_Train_0[:,0]).astype(int)
test_indices = (Y_Test_0[:,0]).astype(int)

trn_idx = train_indices
test_idx = test_indices

X_Train, Y_Train, Y_gc_Train = get_gp_data(X_Train_0, Y_Train_0[:,-1], method_number)
X_Test, Y_Test, Y_gc_Test = get_gp_data(X_Test_0, Y_Test_0[:,-1], method_number)

# X_Train = X_new[trn_idx,:]
# X_Test = X_new[test_idx,:]
# Y_Train = Y_new[trn_idx,:]
# Y_Test = Y_new[test_idx,:]
# Y_gc_Train = Y_gc_new[trn_idx,:]
# Y_gc_Test = Y_gc_new[test_idx,:]

train_data = np.concatenate((X_Train, Y_Train), axis = 1)
test_data = np.concatenate((X_Test, Y_Test), axis = 1)

if method_number == 2:
    data_names =  data_names[:1] + [data_names[-1] + " Discrepancy"]
if method_number == 3:
    data_names =  data_names[:-1] + [data_names[-1] + " Discrepancy"]

train_df = pd.DataFrame(train_data, columns = data_names)
test_df = pd.DataFrame(test_data, columns = data_names)

#Save training and testing data
save_path = "Final_Results/" + code + "/" + gpConfig['SaveName']
os.makedirs(save_path, exist_ok = True)
train_df.to_csv(save_path + "/train_data.csv", index= False)
test_df.to_csv(save_path + "/test_data.csv", index= False)

# Normalize
X_Train_N=X_Train.copy()
X_Test_N=X_Test.copy()
Y_Train_N=Y_Train.copy()
Y_gc_Train_N=Y_gc_Train.copy()
if featureNorm is not None:
    X_Train_N,skScaler_X=normalize(X_Train,method=featureNorm)
    X_Test_N,__=normalize(X_Test,method=featureNorm,skScaler=skScaler_X)
if labelNorm is not None:
    Y_Train_N,skScaler_Y=normalize(Y_Train,method=labelNorm)
    Y_gc_Train_N,__=normalize(Y_gc_Train,method=labelNorm, skScaler=skScaler_Y)

args = (X_Train_N,Y_Train_N, gpConfig)
retrain_count = 0
model, best_min_loss, fit_success, cond_num, trained_hyperparams, model_pretrain, sc_y_scale = \
    train_gp(X_Train_N, Y_Train_N, gpConfig, code, skScaler_Y.scale_, featureNorm, retrain_GP, retrain_count)

best_lml = -1 * best_min_loss
print(best_lml, fit_success, cond_num, trained_hyperparams, sc_y_scale)

# Save the model summary to a CSV file
model_file_name = str(save_path +'/model_summary.txt')
with open(model_file_name, 'w') as file:
    val = gpflow.utilities.read_values(model)
    file.write(str(val))
    file.write("\n Condition Number: " + str(cond_num))
    file.write("\n Fit Success?: " + str(fit_success))
    file.write("\n Log-marginal Likelihood: " + str(best_lml))

# # Get GP predictions
Y_Train_Pred_N,Y_Train_Var_N=gpPredict(model,X_Train_N)
Y_Test_Pred_N,Y_Test_Var_N=gpPredict(model,X_Test_N)

# # Unnormalize
Y_Train_Pred=Y_Train_Pred_N.copy()
Y_Test_Pred=Y_Test_Pred_N.copy()
Y_Train_Var=Y_Train_Var_N.copy()
Y_Test_Var=Y_Test_Var_N.copy()
if labelNorm is not None:
    Y_Train_Pred,__=normalize(Y_Train_Pred_N,skScaler=skScaler_Y,
                            method=labelNorm,reverse=True)
    Y_Test_Pred,__=normalize(Y_Test_Pred_N,skScaler=skScaler_Y,
                            method=labelNorm,reverse=True)
    Y_Train_Var = (skScaler_Y.scale_**2)*Y_Train_Var
    Y_Test_Var = (skScaler_Y.scale_**2)*Y_Test_Var

# #Get data in from such that Y train and Y test are the actual propery predictions
if method_number in [2,3]:
    Y_Test_Pred_plt = Y_Test_Pred + Y_gc_Test #discrepancy_to_property(method_number, Y_Test_Pred, Y_gc, test_indices)
    Y_Train_Pred_plt = Y_Train_Pred + Y_gc_Train #discrepancy_to_property(method_number, Y_Train_Pred, Y_gc, train_indices)
    Y_Test_plt = Y_Test + Y_gc_Test #discrepancy_to_property(method_number, Y_Test, Y_gc, test_indices)
    Y_Train_plt = Y_Train + Y_gc_Train #discrepancy_to_property(method_number, Y_Train, Y_gc, train_indices)
else:
    Y_Test_Pred_plt = Y_Test_Pred  
    Y_Train_Pred_plt = Y_Train_Pred
    Y_Test_plt = Y_Test 
    Y_Train_plt = Y_Train

Y_Test_CI_plt = 1.96*np.sqrt(Y_Test_Var)
Y_Train_CI_plt = 1.96*np.sqrt(Y_Train_Var)

count_CI = count_outside_95(Y_Train_plt, Y_Test_plt,
                 Y_Train_Pred_plt, Y_Test_Pred_plt, 
                 Y_Train_CI_plt, Y_Test_CI_plt)
count_CI = np.array(count_CI)

dir_root = "Final_Results/" + code
os.makedirs(dir_root, exist_ok=True)

np.savetxt(dir_root+f"/{code}_count_CI.txt", count_CI)
np.savetxt(dir_root+f"/{code}_train_indices.txt", trn_idx)
np.savetxt(dir_root+f"/{code}_test_indices.txt", test_idx)
np.savetxt(dir_root+f"/{code}_Y_train_true.txt", Y_Train_plt)
np.savetxt(dir_root+f"/{code}_Y_test_true.txt", Y_Test_plt)
np.savetxt(dir_root+f"/{code}_Y_train_pred.txt", Y_Train_Pred_plt)
np.savetxt(dir_root+f"/{code}_Y_test_pred.txt", Y_Test_Pred_plt)
np.savetxt(dir_root+f"/{code}_Y_train_pred_95CI.txt", Y_Train_CI_plt)
np.savetxt(dir_root+f"/{code}_Y_test_pred_95CI.txt", Y_Test_CI_plt)

if method_number == 2 or method_number == 3:
    parity_plot(code, Y_Train_plt, Y_Test_plt, Y_Train_Pred_plt, Y_Test_Pred_plt, Y_Train_CI_plt, Y_Test_CI_plt,
               gpConfig, save_plot, disc = True)
else:
    parity_plot(code, Y_Train_plt, Y_Test_plt, Y_Train_Pred_plt, Y_Test_Pred_plt, Y_Train_CI_plt, Y_Test_CI_plt,
               gpConfig, save_plot, disc = False)
    
if method_number != 2 and method_number != 3:
    parity_plot_final(code, X, test_idx, trn_idx, Y_Train_plt, Y_Test_plt, Y_Train_Pred_plt, Y_Test_Pred_plt, 
                  gpConfig, save_plot)
    
# Print elapsed time
tf=time.time()
print('Time elapsed: '+'{:.2f}'.format(tf-ti)+' s')

