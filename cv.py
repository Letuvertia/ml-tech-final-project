from train import train

import os
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from options import ArgumentParser
import argparse
from utils import DataManager, get_revenue_pair, get_label_pair, write_test
from evaluate import Grader
from model import ModelWrapper, CancelModel
import yaml
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description='Hotel Booking Demands Problem')
    parser.add_argument('--random_seed', default=1126, type=int)
    parser.add_argument('--tra_path', default='data/train.csv', type=str)
    parser.add_argument('--tst_path', default='data/test.csv', type=str)
    parser.add_argument('--config', default='config/base.yaml', type=str)
    parser.add_argument('--train_task', type=str, required=True)
    parser.add_argument('--can_model', type=str)
    parser.add_argument('--reg_model', type=str)
    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--can_ckpt', type=str)
    parser.add_argument('--save_path', type=str, default='cer_attr/train_other')
    parser.add_argument('--save_name', type=str, default='attr_select.csv')
    
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    pseudo_random_list = np.random.permutation(640)
    val_size = 640 // args.fold_num

    DataMgr = DataManager(args.tra_path, config['base']['drop_list'])
    X_all = DataMgr.get_feat()
    X_all = [X_all[i] for i in pseudo_random_list] # shuffle

    # cross validation
    cer_acc_list = []
    cer_0rcl_list = []
    cer_1rcl_list = []
    for i in range(args.fold_num):
        if i == 0:
            X_val = X_all[: (i+1) * val_size]
            X_tra = X_all[(i+1)*val_size:]
        else:
            X_val = X_all[i * val_size: (i+1) * val_size]
            X_tra = X_all[:i * val_size] + X_all[(i+1) * val_size:]
        
        _, cer_acc, cer_0_recall, cer_1_recall = train(args, config, X_tra, X_val)
        cer_acc_list.append(cer_acc)
        cer_0rcl_list.append(cer_0_recall)
        cer_1rcl_list.append(cer_1_recall)
    org_acc_err = np.mean(cer_acc_list)
    org_0rcl_err = np.mean(cer_0rcl_list)
    org_1rcl_err = np.mean(cer_1rcl_list)

    print(org_acc_err, org_0rcl_err, org_1rcl_err)
    
    
    import csv
    with open(os.path.join(args.save_path, args.save_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['attribute', 
                         'acc error change',
                         'acc att',
                         'acc org',
                         '0 recall error change',
                         '0 recall att',
                         '0 recall org',
                         '1 recall error change',
                         '1 recall att',
                         '1 recall org',])
    
        plugging_attr = set(DataMgr.df.columns) - set(config['base']['drop_list']) - set(['revenue', 'is_canceled', 'adr'])
        for attr_idx, attr in enumerate(plugging_attr):
            print("=========================")
            print(f'Now plugging attribute {attr} ({attr_idx+1}/{len(plugging_attr)})')
            print("=========================")
            DataMgr = DataManager(args.tra_path, config['base']['drop_list'] + [attr])
            X_all = DataMgr.get_feat()
            X_all = [X_all[i] for i in pseudo_random_list]

            cer_acc_list = []
            cer_0rcl_list = []
            cer_1rcl_list = []
            for i in range(args.fold_num):
                if i == 0:
                    X_val = X_all[: (i+1) * val_size]
                    X_tra = X_all[(i+1)*val_size:]
                else:
                    X_val = X_all[i * val_size: (i+1) * val_size]
                    X_tra = X_all[:i * val_size] + X_all[(i+1) * val_size:]
                
                _, cer_acc, cer_0_recall, cer_1_recall = train(args, config, X_tra, X_val)
                cer_acc_list.append(cer_acc)
                cer_0rcl_list.append(cer_0_recall)
                cer_1rcl_list.append(cer_1_recall)
            print('{:} : {:.4f}'.format(attr, (np.mean(cer_acc_list) - org_acc_err) / org_acc_err))
            writer.writerow([attr, 
                             '{:.4f}'.format((np.mean(cer_acc_list) - org_acc_err) / org_acc_err), 
                             '{:.4f}'.format(np.mean(cer_acc_list)), '{:.4f}'.format(org_acc_err),
                             '{:.4f}'.format((np.mean(cer_0rcl_list) - org_0rcl_err) / org_0rcl_err),
                             '{:.4f}'.format(np.mean(cer_0rcl_list)), '{:.4f}'.format(org_0rcl_err),
                             '{:.4f}'.format((np.mean(cer_1rcl_list) - org_1rcl_err) / org_1rcl_err),
                             '{:.4f}'.format(np.mean(cer_1rcl_list)), '{:.4f}'.format(org_1rcl_err)])
            
if __name__ == "__main__":
    main()
