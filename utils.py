from os import makedirs, remove
from os.path import exists, join, basename, dirname

def str_to_bool(v):
    """ This function turn all the True-intended string into True.

    Usage example:
        add_argument('--option1', type=str_to_bool, nargs='?', const=True, default=False, help='...')
    In cmd, all the following lines will be interpreted as option1=True:
        python train.py --option1        # this is set by (nargs='?', const=True)
        python train.py --option1 yes
        python train.py --option1 true
        python train.py --option1 t
        python train.py --option1 y
        python train.py --option1 1
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, file, is_best, is_final=False, is_training=True):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    if is_training:
        torch.save(state, file)

    prefix = ''
    if is_final:
        prefix = 'best_'
    elif is_best:
        prefix = 'final_'
    torch.save(state, join(model_dir, prefix+model_fn))
