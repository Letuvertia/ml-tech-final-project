from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import DataLoader

from options import ArgumentParser
from data_loader import CancelDataset
from model.cancel_toy_model import CancelModel


"""


"""

# Argument parsing
args,arg_groups = ArgumentParser().parse()
print('Training {} model. Arguments are as follows.'.format(arg_groups['base']['model']))
print(args)
print(arg_groups)

# Seed
np.random.seed(arg_groups['base']['seed'])

if arg_groups['base']['model'] == 'cancel':
    # dataset
    dataset = CancelDataset(feature_file='train.csv',
                            label_file='train_label.csv',
                            **arg_groups['dataset'])
    data_loader = DataLoader(dataset, batch_size=arg_groups['cancel']['batch_size'])

    checkpoint_name = os.path.join(arg_groups['base']['result_model_dir'],
                                arg_groups['base']['result_model_fn'] + '.pth.tar')
    print(checkpoint_name)

    # model
    model = CancelModel(save_model_name=checkpoint_name, 
                        seed=arg_groups['base']['seed'],
                        **arg_groups['cancel'])
    model.train_model(train_data_loader=data_loader, val_data_loader=data_loader)
    model.test_model(test_data_loader=data_loader)
    model.save_model()
