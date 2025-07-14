import numpy as np

def APE(data_test, pred_test, data_trn, pred_trn):
    ape_test = np.mean((pred_test - data_test)/data_test) * 100
    ape_trn = np.mean((pred_trn - data_trn)/data_trn) * 100
    return ape_test, ape_trn

data_test = np.loadtxt("Vc_Y_test_true.txt")
pred_test = np.loadtxt("Vc_Y_test_pred.txt")
gc_test = np.loadtxt("Vc_Y_gc_test.txt")
data_trn = np.loadtxt("Vc_Y_train_true.txt")
pred_trn = np.loadtxt("Vc_Y_train_pred.txt")
gc_trn = np.loadtxt("Vc_Y_gc_train.txt")


vc_ape_gcgp = APE(data_test, pred_test, data_trn, pred_trn)
vc_ape_gc_only = APE(data_test, gc_test, data_trn, gc_trn)
vc_ape = np.array([vc_ape_gcgp, vc_ape_gc_only]).flatten()

np.savetxt("vc_ape.txt", vc_ape)