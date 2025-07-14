# -*- coding: utf-8 -*-
"""
Script to train a GP on physicochemical properties.

Last edit: 2024-09-01
Contributors: Barnabas Agbodekhe, Montana Carlozo, Dinis Abranches
"""


import os
import warnings
import time

# Specific
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
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
    2: Y = GP(Y_gc, K(Mw))
    3: Y = GP(Y_gc, K(Mw, Y_gc))
    4: Y = GP(AMw + BY_gc + c, K(Mw, Y_gc))
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
        gpConfig['mean_function']='Custom'
        gpConfig['Name']='y_exp = GP(y_GC, K(x1))'
        gpConfig['SaveName']='model_2'
    if method_number == 3:
        gpConfig['mean_function']='Constant'
        gpConfig['Name']='y_exp = GP(y_GC, K(x1,x2))'
        gpConfig['SaveName']='model_3'
    if method_number == 4:
        gpConfig['mean_function']='Linear'
        gpConfig['Name']='y_exp = GP(B@X, K(x1,x2))'
        gpConfig['SaveName']='model_4'
    else: 
        if method_number not in [1 , 2, 3, 4]:
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
    Y_gp = Y
    Y_gp = Y_gp.reshape(-1,1)
    Y_gc = X[:,1].reshape(-1,1)
    return X_gp, Y_gp, Y_gc



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
    if method == 'None':
        inputArray=inputArray
        skScaler = None
    else:
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



class CustomMeanFunction(gpflow.mean_functions.MeanFunction):
    def __init__(self, X_train, Z_train):
        super().__init__()
        self.X_train = X_train
        self.Z_train = Z_train
        self.X_test = None
        self.Z_test = None

    def update_Z_test(self, X_test, Z_test):
        """Update the mean function with the test data."""
        self.X_test = X_test
        self.Z_test = Z_test

    def __call__(self, X):
        # Determine whether we are in the training or testing phase
        if self.X_test is not None and np.array_equal(X, self.X_test):
            # In testing phase, return Z_test values
            return tf.convert_to_tensor(self.Z_test, dtype=tf.float64)
        elif np.array_equal(X, self.X_train):
            # In training phase, return Z_train values
            return tf.convert_to_tensor(self.Z_train, dtype=tf.float64)
        else:
            raise ValueError("The provided X values do not match the stored training or test X values.")        
        
        
        
        

# Build GPR model function with bounded hyperparameters
def build_model_with_bounded_params(X, Y, Y_gc, kern, low, high, \
                                    high_alpha, init_val1, init_val2, init_val3, \
                                    useWhite, trainLikelihood, anisotropic, typeMeanFunc):
    """
    build_model_with_bounded_params(*) creates a GP model object with bounded hyperparameters and initial 
    values

    Parameters
    ----------
    X : numpy array
        Feature data
    Y : numpy array
        Label data
    Y_gc: numpy array
        GC predictions for use in custom mean function
    low : float
        lower bound on all hyperparameters
    high : float
        upper bound on all hyperparameters except alpha for the RQ kernel
    high_alpha : float
        upper bound on alpha hyperparameter for the RQ kernel
    init_val1 : float
        initial values for first length scale and alpha parameter
    init_val2 : float
        initial values for second length scale for anisotropic kernels
        for isotropic kernels, only one initial value (init_val1) is used
    init_val1 : float
        initial values for variance or scale hyperparameter of kernel 1 (not Whitenoise kernel)
    
    Returns
    -------
    model : Gpflow model object
        GP model object with bounded hyperparameters and initial values
        
    """
    
    low = tf.cast(low, dtype=tf.float64)
    high = tf.cast(high, dtype=tf.float64)
    high_alpha = tf.cast(high_alpha, dtype=tf.float64)
    init_val1 = tf.cast(init_val1, dtype=tf.float64)
    init_val2 = tf.cast(init_val2, dtype=tf.float64)
    init_val3 = tf.cast(init_val3, dtype=tf.float64)
    if anisotropic == True:
        lsc = gpflow.Parameter([init_val1, init_val2], transform=tfb.Sigmoid(low , high), dtype=tf.float64)
    else:
        lsc = gpflow.Parameter(init_val1, transform=tfb.Sigmoid(low , high), dtype=tf.float64)
    alf = gpflow.Parameter(init_val1, transform=tfb.Sigmoid(low , high_alpha), dtype=tf.float64)
    var = gpflow.Parameter(init_val3, transform=tfb.Sigmoid(low , high), dtype=tf.float64)
    if kern == "RQ":
        kernel_ = gpflow.kernels.RationalQuadratic()
        kernel_.alpha = alf
        kernel_.lengthscales = lsc 
        kernel_.variance = var
    elif kern == "RBF":
        kernel_ = gpflow.kernels.RBF()
        kernel_.lengthscales = lsc
        kernel_.variance = var
    elif kern == "Matern12":
        kernel_ = gpflow.kernels.Matern12()
        kernel_.lengthscales = lsc
        kernel_.variance = var
    elif kern == "Matern32":
        kernel_ = gpflow.kernels.Matern32()
        kernel_.lengthscales = lsc
        kernel_.variance = var
    elif kern == "Matern52":
        kernel_ = gpflow.kernels.Matern52()
        kernel_.lengthscales = lsc
        kernel_.variance = var
    if useWhite == True:
        #white_var = np.array(np.random.uniform(0.05, 1.0))
        final_kernel = kernel_+gpflow.kernels.White(variance=1.0)
    else:
        final_kernel = kernel_
        
    if typeMeanFunc == 'Zero':
        mf = None
    if typeMeanFunc == 'Custom':
        mf = CustomMeanFunction(X, Y_gc)
    if typeMeanFunc == 'Constant':
        #If constant value is selected but no value is given, default to zero mean
        mf_val = np.array([0,1]).reshape(-1,1)
        mf = gpflow.functions.Linear(mf_val)
    if typeMeanFunc == 'Linear':
        A = np.ones((X.shape[1],1))
        mf = gpflow.functions.Linear(A)
    model_ = gpflow.models.GPR(data=(X, Y), kernel=final_kernel, mean_function=mf, noise_variance=10**-5)
    if typeMeanFunc == 'Constant':
        gpflow.set_trainable(model_.mean_function.A, False)
        gpflow.set_trainable(model_.mean_function.b, False)
    gpflow.utilities.set_trainable(model_.likelihood.variance,trainLikelihood)
    model = model_
    return model



def buildGP(X_Train, Y_Train, Y_gc, gpConfig, code, featurenorm, retrain_count):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features
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
    sc_y_scale : Scikit learn standard scaler object
        standard scaler fitted on label training data 
    retrain_count : int
        Current GP retrain number

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RQ')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    typeMeanFunc=gpConfig.get('mean_function','Zero')
    opt_method=gpConfig.get('opt_method','L-BFGS-B')
    anisotropy=gpConfig.get('anisotropic','False')
    
    seed_ = int(retrain_count) * 100
    np.random.seed(seed_)
    tf.random.set_seed(seed_)
    
    if retrain_count == 0:
        init_val1 = 1
        init_val2 = 1
        init_val3 = 1
    else:
        init_val1 = np.array(np.random.uniform(0, 100))
        init_val2 = np.array(np.random.uniform(0, 100))
        init_val3 = np.array(np.random.lognormal(0, 1.0))
    
    low_bound = 1e-5
    up_bound = 1e2
    alpha_up_bound = 5e3
    
    model = build_model_with_bounded_params(X_Train, Y_Train, Y_gc, kernel, low_bound, up_bound, alpha_up_bound,\
                                            init_val1, init_val2, init_val3, useWhiteKernel, \
                                            trainLikelihood, anisotropy, typeMeanFunc)
    model_pretrain = deepcopy(model)
    # print(gpflow.utilities.print_summary(model))
    condition_number = np.linalg.cond(model.kernel(X_Train))
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
        
    return model, aux, condition_number, obj_func, opt_success, retrain_count, model_pretrain



def train_gp(X_Train, Y_Train, Y_gc, gpConfig, code, sc_y, featurenorm, retrain_GP, retrain_count):
    """
    Trains the GP given training data.
    
 
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
            buildGP(X_Train, Y_Train, Y_gc, gpConfig, code, featurenorm, retrain_count)
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
    
    if sc_y != None:
        sc_y_scale = sc_y.scale_
    else:
        sc_y_scale = None

    return best_model,best_minimum_loss,best_model_success,best_condition_num,trained_hyperparams,best_model_pretrain,sc_y_scale


def gpPredict(model,X):
    """
    gpPredict() returns the prediction and variance of the GP model
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
    VAR=GP_Var
    # Output
    return Y,VAR


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mapd = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = np.mean(np.abs((y_true - y_pred)))
    return r2, mapd, mae


    


# =============================================================================
#######################################

# GP training and property predictions

#######################################
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================
#Model data is found based on method number

dbPath=""
# Property Code
code='Vc' # 'Hvap', 'Vc', 'Pc', 'Tc', 'Tb', 'Tm'
# Other placeholders
kernel = 'Matern32'  #Other Options: RQ, RBF, Matern12, Matern32, Matern52
anisotropic = False
method_number = 4
# Define normalization methods
featureNorm='Standardization' # None,Standardization,MinMax
labelNorm='Standardization' # None,Standardization,LogStand
opt_method = 'L-BFGS-B' #Other Options: L-BFGS-B, BFGS
useWhiteKernel = True
trainLikelihood = False
save_plot = True 
retrain_GP = 10



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
db=pd.read_csv(os.path.join(dbPath,code+'_prediction_data_fcl.csv'))
db=db.dropna()
X=db.iloc[:,2:-1].copy().to_numpy('float')
data_names=db.columns.tolist()[2:]
Y=db.iloc[:,-1].copy().to_numpy('float')
Y = Y.reshape(-1,1)
Y_gc = X[:,-1].reshape(-1,1)
MW = X[:,-2].reshape(-1,1)

# >>>>>>>>>  Stratification based on features <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

X_data = db.iloc[:,2:-1].copy()
num_rows_X = X_data.shape[0]  
y_data_dum = (np.ones((num_rows_X, 2))).astype(int)
indices = np.arange(X_data.shape[0])
y_stratify = np.column_stack((indices, y_data_dum))
X_stratify = X_data.values

X_ = np.array(y_stratify)
y_ = np.array(X_stratify)
y_strat = y_ 
X_strat = X_ 

seed = 42
np.random.seed(seed)

X_Train_0, y_Train_0, X_valTest_0, y_valTest_0 = iterative_train_test_split(X_strat, y_strat, test_size = 0.2)

train_indices = (X_Train_0[:,0]).astype(int)
test_indices = (X_valTest_0[:,0]).astype(int)


trn_idx = train_indices
test_idx = test_indices

X_Train_0 = X[trn_idx, :]
X_Test_0 = X[test_idx, :]
Y_Train_0 = Y[trn_idx, :]
Y_Test_0 = Y[test_idx, :]


X_Train, Y_Train, Y_gc_Train = get_gp_data(X_Train_0, Y_Train_0[:,-1], method_number)
X_Test, Y_Test, Y_gc_Test = get_gp_data(X_Test_0, Y_Test_0[:,-1], method_number)

train_data = np.concatenate((X_Train, Y_Train), axis = 1)
test_data = np.concatenate((X_Test, Y_Test), axis = 1)

if method_number == 2:
    data_names =  [data_names[0], data_names[-1]]


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
Y_gc_Test_N=Y_gc_Test.copy()
if featureNorm is not None:
    X_Train_N,skScaler_X=normalize(X_Train,method=featureNorm)
    X_Test_N,__=normalize(X_Test,method=featureNorm,skScaler=skScaler_X)
else:
    skScaler_X = None
if labelNorm is not None:
    Y_Train_N,skScaler_Y=normalize(Y_Train,method=labelNorm)
    Y_gc_Train_N,__=normalize(Y_gc_Train,method=labelNorm, skScaler=skScaler_Y)
    Y_gc_Test_N,__=normalize(Y_gc_Test,method=labelNorm, skScaler=skScaler_Y)
else:
    skScaler_Y = None

args = (X_Train_N,Y_Train_N, gpConfig)
retrain_count = 0
model, best_min_loss, fit_success, cond_num, trained_hyperparams, model_pretrain, sc_y_scale = \
    train_gp(X_Train_N, Y_Train_N, Y_gc_Train_N, gpConfig, code, skScaler_Y, featureNorm, retrain_GP, retrain_count)

best_lml = -1 * best_min_loss
best_lml = best_lml.numpy()
print(best_lml, fit_success, cond_num, trained_hyperparams, sc_y_scale)
print()
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

if method_number == 2:
    model.mean_function.update_Z_test(X_Test_N, Y_gc_Test_N)

Y_Test_Pred_N,Y_Test_Var_N=gpPredict(model,X_Test_N)

# # Unnormalize
Y_Train_Pred=Y_Train_Pred_N.copy()
Y_Test_Pred=Y_Test_Pred_N.copy()
Y_Train_Var=Y_Train_Var_N.copy()
Y_Test_Var=Y_Test_Var_N.copy()
if labelNorm != 'None':
    Y_Train_Pred,__=normalize(Y_Train_Pred_N,skScaler=skScaler_Y,
                            method=labelNorm,reverse=True)
    Y_Test_Pred,__=normalize(Y_Test_Pred_N,skScaler=skScaler_Y,
                            method=labelNorm,reverse=True)
    Y_Train_Var = (skScaler_Y.scale_**2)*Y_Train_Var
    Y_Test_Var = (skScaler_Y.scale_**2)*Y_Test_Var


Y_Test_Pred_plt = Y_Test_Pred  
Y_Train_Pred_plt = Y_Train_Pred
Y_Test_plt = Y_Test 
Y_Train_plt = Y_Train

    
    
r2_train, mapd_train, mae_train = evaluate(Y_Train_plt, Y_Train_Pred_plt)
r2_test, mapd_test, mae_test = evaluate(Y_Test_plt, Y_Test_Pred_plt)


# Save the model summary to a txt file
model_file_name = str(f'model_summary_{code}_{method_number}_{kernel}_{anisotropic}.txt')
with open(model_file_name, 'w') as file:
    val = gpflow.utilities.read_values(model)
    file.write(str(val))
    file.write("\n Label standard scaler scale parameter: " + str(sc_y_scale))
    file.write("\n Condition Number: " + str(cond_num))
    file.write("\n Fit Success?: " + str(fit_success))
    file.write("\n Test MAPD: " + str(mapd_test))
    file.write("\n Train MAPD: " + str(mapd_train))
    file.write("\n Test MAE: " + str(mae_test))
    file.write("\n Train MAE: " + str(mae_train))
    file.write("\n Test R2: " + str(r2_test))
    file.write("\n Train R2: " + str(r2_train))
    file.write("\n Log-marginal Likelihood: " + str(best_lml))


  
tf=time.time()
print('Time elapsed: '+'{:.2f}'.format(tf-ti)+' s')




