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
import joblib

def main():
    parser = argparse.ArgumentParser(
        description='Hotel Booking Demands Problem')
    parser.add_argument('--random_seed', default=7, type=int)
    parser.add_argument('--tra_path', default='data/train.csv', type=str)
    parser.add_argument('--tst_path', default='data/test.csv', type=str)
    parser.add_argument('--config', default='config/base.yaml', type=str)
    parser.add_argument('--train_task', type=str, required=True)
    parser.add_argument('--use_onehot', action='store_true')
    parser.add_argument('--can_model', type=str)
    parser.add_argument('--reg_model', type=str)
    parser.add_argument('--fold_num', type=int)
    parser.add_argument('--can_ckpt', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    pseudo_random_list = np.random.permutation(640)
    val_size = 640 // args.fold_num

    DataMgr = DataManager(args.tra_path, args.tst_path)
    X_all_can = DataMgr.get_train_feat(config['base']['cancel_drop_list'], use_onehot=joblib.load(args.can_ckpt).use_onehot)
    X_all_rev = DataMgr.get_train_feat(config['base']['target_drop_list'], use_onehot=args.use_onehot)
    
    X_all_rev = [X_all_rev[i] for i in pseudo_random_list]
    X_all_can = [X_all_can[i] for i in pseudo_random_list]

    rev_list = []
    mae_list = []
    for i in range(args.fold_num):
        if i == 0:
            X_tra = X_all_rev[(i+1)*val_size:]
            X_val = (X_all_rev[: (i+1) * val_size],
                     X_all_can[: (i+1) * val_size])
        else:
            X_tra = X_all_rev[:i * val_size] + X_all_rev[(i+1) * val_size:]
            X_val = (X_all_rev[i * val_size: (i+1) * val_size],
                     X_all_can[i * val_size: (i+1) * val_size])

        _, rev, mae = train(args, config, X_tra, X_val,
                            DataMgr.get_input_dim(), DataMgr.get_onehot_dim())
        print(rev, mae)
        rev_list.append(rev)
        mae_list.append(mae)
    org_rev = np.mean(rev_list)
    org_mae = np.mean(mae_list)
    print(org_rev, org_mae)
    # import csv
    # rev_name = 'rev_attr_select3.csv'
    # with open(rev_name, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow(['rev', '{:.4f}'.format(org_rev)])
    #     writer.writerow(['mae', '{:.4f}'.format(org_mae)])
    #     writer.writerow(['attribute', 'rev change', 'mae change', 'rev', 'mae'])

    # plugging_attr = sorted(list(set(pd.read_csv(args.tra_path).columns) -
    #                             set(config['base']['target_drop_list']) -
    #                             set(['revenue', 'is_canceled', 'adr'])))

    # DataMgr = DataManager(args.tra_path)
    # for attr in plugging_attr:
    #     X_all_rev = DataMgr.get_feat(
    #         config['base']['target_drop_list'] + [attr])
    #     X_all_rev = [X_all_rev[i] for i in pseudo_random_list]

    #     rev_list = []
    #     mae_list = []

    #     for i in range(args.fold_num):
    #         if i == 0:
    #             X_tra = X_all_rev[(i+1)*val_size:]
    #             X_val = (X_all_rev[: (i+1) * val_size],
    #                      X_all_can[: (i+1) * val_size])
    #         else:
    #             X_tra = X_all_rev[:i * val_size] + X_all_rev[(i+1) * val_size:]
    #             X_val = (X_all_rev[i * val_size: (i+1) * val_size],
    #                      X_all_can[i * val_size: (i+1) * val_size])

    #         _, rev, mae = train(args, config, X_tra, X_val,
    #                             DataMgr.get_input_dim(), DataMgr.get_onehot_dim())
    #         rev_list.append(rev)
    #         mae_list.append(mae)

    #     print('rev: {:} : {:.4f}'.format(
    #         attr, (org_rev - np.mean(rev_list)) / org_rev), '| mae: {:} : {:.4f}'.format(
    #         attr, (org_mae - np.mean(mae_list)) / org_mae))

    #     with open(rev_name, 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         writer.writerow([attr, '{:.4f}'.format(
    #             (org_rev - np.mean(rev_list)) / org_rev), '{:.4f}'.format((org_mae - np.mean(mae_list)) / org_mae), '{:.4f}'.format(np.mean(rev_list)), '{:.4f}'.format(np.mean(mae_list))])


if __name__ == "__main__":
    main()
