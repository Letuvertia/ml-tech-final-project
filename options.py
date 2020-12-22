import argparse
from utils import str_to_bool


class ArgumentParser():
    def __init__(self, model=None):
        self.parser = argparse.ArgumentParser(description='Hotel Booking Demands Problem')
        self.add_base_parameters()
        self.add_dataset_parameters()
        self.universal_arguments = {'base', 'dataset'}

        # parameters for random forest
        self.add_rf_parameters()

        # parameters for SVM
        self.add_svm_parameters()
        
        # parameters for adaptive boost
        self.add_adaboost_parameters()
        
        # parameters for neural networks
        self.add_nn_parameters()
        
    
    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        # add the arguments that will be used by all models
        base_params.add_argument('--model', type=str, default='', help='type of model to use')


    def add_dataset_parameters(self):
        dataset_params = self.parser.add_argument_group('dataset')
        dataset_params.add_argument('--dataset-csv-path', type=str, default='data', help='path to dataset csv folder')

    
    # example function of how to add your arguments
    def add_example_parameters(self):
        example_params = self.parser.add_argument_group('example')
        example_params.add_argument('--result-model-fn', type=str, default='trained_model_filename', help='trained model filename')
        example_params.add_argument('--result-model-dir', type=str, default='trained_model', help='path to trained models folder')


    def add_rf_parameters(self):
        rf_params = self.parser.add_argument_group('rf')
        # add your rf params
        rf_params.add_argument('--rf_param1', type=float, default=1., help='...')
    

    def add_svm_parameters(self):
        svm_params = self.parser.add_argument_group('svm')
        # add your svm params
        svm_params.add_argument('--svm_param1', type=float, default=1., help='...')
    

    def add_adaboost_parameters(self):
        adaboost_params = self.parser.add_argument_group('adaboost')
        # add your adaboost params
        adaboost_params.add_argument('--ada_param1', type=float, default=1., help='...')
    

    def add_nn_parameters(self):
        nn_params = self.parser.add_argument_group('nn')
        # add your nn params
        nn_params.add_argument('--nn_param1', type=float, default=1., help='...')


    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())
        arg_groups = {}
        for group in self.parser._action_groups:
            if group.title == args.model or group.title in self.universal_arguments:
                group_dict = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}
                arg_groups[group.title] = group_dict
        return (args, arg_groups)


if __name__ == '__main__':
    """ options.ArgumentParser() usage example

    Note that the dictionary 'arg_groups' only contains the base arguments and \
        the arguments of the given model (depend on --model)
    try to run 'python options.py --model svm'
    """
    args,arg_groups = ArgumentParser().parse()
    print('args: ', args)
    print('arg_groups: ', arg_groups)
    print(arg_groups['base']['model'])
