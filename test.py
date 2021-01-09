from __future__ import print_function, division
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
from utils import DataManager, get_revenue_pair, get_label_pair, write_test
from evaluate import Grader
from model import ModelWrapper, CancelModel
import yaml
import joblib
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""


"""

def get_subset(Y_pre, X, assign_labels): 
    X_sub = []
    idx = []
    for i in range(len(Y_pre)):
        if Y_pre[i] in assign_labels:
            X_sub.append(X[i])
            idx.append(i)
    return X_sub, idx


class Visualization:
    def __init__(self, y_true, y_pred, target_names=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.target_names = target_names
        self.counter = 0

    def confusion_matrix(self, name):
     
        cm = confusion_matrix(self.y_true, self.y_pred)
        df_cm = pd.DataFrame(cm)
        if self.target_names is not None:
            df_cm.columns = [name for name in self.target_names]
            df_cm.index = [name for name in self.target_names]
        plt.figure(f"{self.counter}. Confusion Matrix")
        ax = sns.heatmap(df_cm, annot=True)
        ax.set(xlabel="Predicted label", ylabel="True label", title="Confustion Matrix")
        plt.yticks(rotation=0)
        plt.tight_layout()
        self.counter += 1
        plt.savefig(name)
        plt.clf()
        return self

    def classification_report(self):
        report = classification_report(
            self.y_true, self.y_pred, target_names=self.target_names, output_dict=True
        )
        for key in ["accuracy", "macro avg", "weighted avg"]:
            report.pop(key, None)
        for key in report:
            report[key].pop("support", None)
        plt.figure(f"{self.counter}. Classification Report")
        ax = sns.heatmap(pd.DataFrame(report).T, annot=True)
        ax.set(title="Classification Report")
        plt.yticks(rotation=0)
        plt.tight_layout()
        self.counter += 1
        return self

    def show(self):
        plt.show()

def output_val(args, model, DataMgr):
    X_all_tar = DataMgr.get_feat(model.drop_list, model.filter_all, model.use_onehot)
    X_all_can = DataMgr.get_feat(model.cancel_model.drop_list, model.cancel_model.filter_all, model.cancel_model.use_onehot)

    X_tra_tar, X_val_tar, X_tra_can, X_val_can = train_test_split(X_all_tar, X_all_can, test_size=args.val_size, random_state=args.random_seed)
    X_val_tar, Y_val = get_label_pair(X_val_tar)
    X_val_can, _ = get_label_pair(X_val_can)
    
    X_val = (X_val_tar, X_val_can)
    Y_val_pre = np.array(model.predict(X_val))
    
    Y_val = np.array(Y_val)
    return X_val, Y_val, Y_val_pre

def output_tst(args, model, DataMgr):
    X_tst_can = DataMgr.get_feat(model.cancel_model.drop_list, model.cancel_model.filter_all, model.cancel_model.use_onehot, False)
    X_tst_tar = DataMgr.get_feat(model.drop_list, model.filter_all, model.use_onehot, False)
    X_tst = (X_tst_tar, X_tst_can)
    Y_pre = np.array(model.predict(X_tst))
    return X_tst_tar, Y_pre

def load_model(args, config, reg_ckpt, can_ckpt=None):
    model = ModelWrapper(args, config, filter_all=True, use_onehot=True)
    model.load(reg_ckpt)    
    if can_ckpt is not None:
        model.cancel_model.load(can_ckpt)    
    return model

def blending(Y_pre_DNR, Y_pre_RFR, label):
    lst = []
    alpha = 0.6
    for i in range(len(Y_pre_RFR)):
        if Y_pre_RFR[i] >= label and Y_pre_DNR[i] <= label:
            Y_pre = Y_pre_RFR[i] * alpha + Y_pre_DNR[i] * (1 - alpha)
            if Y_pre < label:
                lst.append(i)
    return lst

def main():
    parser = argparse.ArgumentParser(
        description='Hotel Booking Demands Problem')
    parser.add_argument('--random_seed', default=1126, type=int)
    parser.add_argument('--val_size', default=0.2, type=float)
    parser.add_argument('--tra_path', default='data/train.csv', type=str)
    parser.add_argument('--tst_path', default='data/test.csv', type=str)
    parser.add_argument('--config', default='config/base.yaml', type=str)
    parser.add_argument('--train_task', type=str, required=True)
    parser.add_argument('--reg_model', default='RFR', type=str)
    parser.add_argument('--use_onehot', action='store_true')
    parser.add_argument('--filter_all', action='store_true')
    parser.add_argument('--can_ckpt', type=str)
    parser.add_argument('--reg_ckpt', type=str)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    DataMgr = DataManager(args.tra_path, args.tst_path)   

    DNR_model = load_model(args, config, 'trained_models/adr/DNR_REV_73.336_MAE_0.211.pkl', 'trained_models/cancel/RFC_best.pkl')
    RFR_model = load_model(args, config, 'trained_models/adr/RFR_REV_77.078_MAE_0.242_new.pkl', 'trained_models/cancel/RFC_best.pkl')
    RFR_model_res1 = load_model(args, config, 'trained_models/adr/RFR_REV_25.503_MAE_0.600.pkl', 'trained_models/cancel/RFC_best.pkl')
    RFR_model_res2 = load_model(args, config, 'trained_models/adr/RFR_REV_38.897_MAE_0.164.pkl', 'trained_models/cancel/RFC_best.pkl')

    # X_val_DNR, Y_val, Y_val_DNR_pre = output_val(args, DNR_model, DataMgr)
    # X_val_RFR, Y_val, Y_val_RFR_pre = output_val(args, RFR_model, DataMgr)

    # label = 3
    # print('label = {:d}'.format(label))
    # Y_val_pre = Y_val_RFR_pre // 1

    # for i in range(len(Y_val)):
    #     if Y_val_RFR_pre[i] >= label and Y_val_DNR_pre[i] <= label:
    #         alpha = 0.6
    #         print(Y_val[i], Y_val_DNR_pre[i] * alpha +  Y_val_RFR_pre[i] * (1 - alpha), Y_val_DNR_pre[i], Y_val_RFR_pre[i])
    #         correct = Y_val_DNR_pre[i] * alpha +  Y_val_RFR_pre[i] * (1 - alpha)
    #         Y_val_pre[i] = correct // 1
 
    # label = 4
    # print('label = {:d}'.format(label))
    # for i in range(len(Y_val)):
    #     if Y_val_RFR_pre[i] >= label and Y_val_DNR_pre[i] <= label:
    #         alpha = 0.6
    #         print(Y_val[i], Y_val_DNR_pre[i] * alpha +  Y_val_RFR_pre[i] * (1 - alpha), Y_val_DNR_pre[i], Y_val_RFR_pre[i])
    #         correct = Y_val_DNR_pre[i] * alpha +  Y_val_RFR_pre[i] * (1 - alpha)
    #         Y_val_pre[i] = correct // 1
 
    # label = 5
    # print('label = {:d}'.format(label))
    # for i in range(len(Y_val)):
    #     if Y_val_RFR_pre[i] >= label and Y_val_DNR_pre[i] <= label:
    #         alpha = 0.6
    #         print(Y_val[i], Y_val_DNR_pre[i] * alpha +  Y_val_RFR_pre[i] * (1 - alpha), Y_val_DNR_pre[i], Y_val_RFR_pre[i])
    #         correct = Y_val_DNR_pre[i] * alpha +  Y_val_RFR_pre[i] * (1 - alpha)
    #         Y_val_pre[i] = correct // 1
 
    # label = 6
    # print('label = {:d}'.format(label))
    # for i in range(len(Y_val)):
    #     if Y_val_RFR_pre[i] >= label and Y_val_DNR_pre[i] <= label:
    #         alpha = 0.6
    #         print(Y_val[i], Y_val_DNR_pre[i] * alpha +  Y_val_RFR_pre[i] * (1 - alpha), Y_val_DNR_pre[i], Y_val_RFR_pre[i])
    #         correct = Y_val_DNR_pre[i] * alpha +  Y_val_RFR_pre[i] * (1 - alpha)
    #         Y_val_pre[i] = correct // 1
 
    # Y_val_RFR_pre_ = Y_val_RFR_pre // 1
    # print(Y_val_RFR_pre_)
    # vis = Visualization(y_true=Y_val, y_pred=Y_val_RFR_pre_)
    # vis.confusion_matrix('RFR_resample.png')
    
    # print(Y_val_pre)
    # vis = Visualization(y_true=Y_val, y_pred=Y_val_pre)
    # vis.confusion_matrix('blending.png')
    
    # thr = 0.065
    # Y_val_pre = Y_val_RFR_pre
    # Y_val_pre[(Y_val_pre - Y_val_pre//1) < thr] = Y_val_pre[(Y_val_pre - Y_val_pre//1) < thr] - (thr + 0.00001)
    # Y_val_pre = Y_val_pre // 1
    # vis = Visualization(y_true=Y_val, y_pred=Y_val_pre)
    # vis.confusion_matrix('shrink.png')
    
    # grader = Grader(X_val)    

    # # revenue MAE a.k.a REV
    # rev = grader.eval_revenue(model)
    # mae = grader.eval_mae(model)
    # cer = grader.eval_cancel_error_rate(model)
    # print(rev, mae)
    
    _, Y_pre_DNR = output_tst(args, DNR_model, DataMgr)
    X_tst_tar, Y_pre_RFR = output_tst(args, RFR_model, DataMgr)
    X_tst_tar, Y_pre_RFR_res1 = output_tst(args, RFR_model_res1, DataMgr)
    X_tst_tar, Y_pre_RFR_res2 = output_tst(args, RFR_model_res2, DataMgr)

    lst_1 = blending(Y_pre_DNR, Y_pre_RFR, 3)
    lst_2 = blending(Y_pre_DNR, Y_pre_RFR, 4)
    lst_3 = blending(Y_pre_DNR, Y_pre_RFR, 5)
    lst_4 = blending(Y_pre_DNR, Y_pre_RFR, 6)
    # lst_5 = blending(Y_pre_DNR, Y_pre_RFR, 7)
    # lst_6 = blending(Y_pre_DNR, Y_pre_RFR, 8)
    # lst_7 = blending(Y_pre_DNR, Y_pre_RFR, 9)
    # print(lst_3, lst_4, lst_5, lst_6, lst_7)
    thr = 0.06
    print(((Y_pre_RFR - Y_pre_RFR//1) < thr).sum())
    print(Y_pre_RFR[(Y_pre_RFR - Y_pre_RFR//1) < thr])
    shrink_lst = [i for i in range(len(Y_pre_RFR)) if (Y_pre_RFR[i] - Y_pre_RFR[i]//1) < thr]
    
    blending_lst = []
    for i in range(1, 4):
        blending_lst = blending_lst + eval('lst_{:d}'.format(i)) 
    
    
    Y_pre = Y_pre_RFR // 1
    for i in range(len(Y_pre)):
        a = Y_pre_RFR[i] // 1
        b = Y_pre_RFR_res1[i] // 1
        c = Y_pre_RFR_res2[i] // 1
        if a != b or b != c or c != a:
            s = 0.35 * (Y_pre_RFR_res1[i] + Y_pre_RFR_res2[i]) + 0.3 * Y_pre_RFR[i]
            s = s // 1
            Y_pre[i] = s
    lst = sorted(set(blending_lst + shrink_lst))
    print(len(lst), lst)
    
    Y_pre[lst] = (Y_pre_RFR // 1)[lst] - 1
    Y_pre[128] = 8.0
    
    write_test(X_tst_tar, Y_pre, 'output_RF_try_l.csv')

if __name__ == "__main__":
    main()
