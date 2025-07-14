import numpy as np

def APE(data_test, pred_test, data_trn, pred_trn):
    ape_test = np.mean(data_test - pred_test)
    ape_trn = np.mean(data_trn - pred_trn)
    return ape_test, ape_trn

data_test = np.loadtxt("Vc_Y_test_true.txt")
pred_test = np.loadtxt("Vc_Y_test_pred.txt")
data_trn = np.loadtxt("Vc_Y_train_true.txt")
pred_trn = np.loadtxt("Vc_Y_train_pred.txt")

vc_ape = APE(data_test, pred_test, data_trn, pred_trn)
vc_ape = np.array([vc_ape]).flatten()

np.savetxt("vc_ape.txt", vc_ape)