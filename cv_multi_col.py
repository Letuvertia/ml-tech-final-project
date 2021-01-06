import argparse
import csv
import numpy as np
import os
import pandas as pd
import random
import yaml

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from evaluate import Grader
from model import ModelWrapper, CancelModel
from options import ArgumentParser
from train import train
from utils import DataManager, get_revenue_pair, get_label_pair, write_test


def main():

    # parsing
    parser = argparse.ArgumentParser(
        description='Hotel Booking Demands Problem')
    parser.add_argument('--random_seed', default=1126, type=int)
    parser.add_argument('--tra_path', default='data/train.csv', type=str)
    parser.add_argument('--tst_path', default='data/test.csv', type=str)
    parser.add_argument('--config', default='config/base.yaml', type=str)
    parser.add_argument('--train_task', type=str, required=True)
    parser.add_argument('--can_model', type=str, default='RanForestC')
    parser.add_argument('--reg_model', type=str)
    parser.add_argument('--fold_num', type=int, default=10)
    parser.add_argument('--can_ckpt', type=str)
    parser.add_argument('--save_path', type=str, default='cer_attr/train_other')
    parser.add_argument('--save_name', type=str, default='attr_select_multicol.csv')
    parser.add_argument('--n_plugged_col', type=int, default=2)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    # cross validation index
    pseudo_random_list = np.random.permutation(640)
    val_size = 640 // args.fold_num

    DataMgr = DataManager(args.tra_path, config['base']['drop_list'])
    X_all = DataMgr.get_feat()
    X_all = [X_all[i] for i in pseudo_random_list] # shuffle

    # cross validation for all columns
    '''
    cer_list = []
    for i in range(args.fold_num):
        if i == 0:
            X_val = X_all[: (i+1) * val_size]
            X_tra = X_all[(i+1)*val_size:]
        else:
            X_val = X_all[i * val_size: (i+1) * val_size]
            X_tra = X_all[:i * val_size] + X_all[(i+1) * val_size:]
        
        _, cer = train(args, config, X_tra, X_val)
        cer_list.append(cer)
    org_err = np.mean(cer_list)
    print(org_err)
    #'''
    org_acc_err = 0.12208936587349739
    org_0rcl_err = 0.06274843108203057
    org_1rcl_err = 0.2313964641440278
    
    # cross validation for removing features
    target_shit_columns = ['is_repeated_guest', 
                           'arrival_date_week_number',
                           'children',
                           'ID']
    shit_col_comb = []
    def find_all_comb(left_col, cur_col, n_left_col=args.n_plugged_col):
        nonlocal shit_col_comb
        for idx, col in enumerate(left_col):
            left_col_co = left_col[idx+1:].copy()
            cur_col_co = cur_col.copy()
            cur_col_co.append(col)
            if n_left_col != 1:
                find_all_comb(left_col_co, cur_col_co, n_left_col-1)
            else:
                shit_col_comb.append(cur_col_co)
    find_all_comb(target_shit_columns, [])
    print('*combinations of to-be-plugged columns:')
    print(shit_col_comb, end='\n')
    #raise ImportError
    

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

        for comb in shit_col_comb:
            DataMgr = DataManager(args.tra_path, config['base']['drop_list'] + comb)
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
            print('{:} : {:.4f}'.format('+'.join(comb), (np.mean(cer_acc_list) - org_acc_err) / org_acc_err))
            writer.writerow(['+'.join(comb), 
                             '{:.4f}'.format((np.mean(cer_acc_list) - org_acc_err) / org_acc_err), 
                             '{:.4f}'.format(np.mean(cer_acc_list)), '{:.4f}'.format(org_acc_err),
                             '{:.4f}'.format((np.mean(cer_0rcl_list) - org_0rcl_err) / org_0rcl_err),
                             '{:.4f}'.format(np.mean(cer_0rcl_list)), '{:.4f}'.format(org_0rcl_err),
                             '{:.4f}'.format((np.mean(cer_1rcl_list) - org_1rcl_err) / org_1rcl_err),
                             '{:.4f}'.format(np.mean(cer_1rcl_list)), '{:.4f}'.format(org_1rcl_err)])
                             
if __name__ == "__main__":
    main()
