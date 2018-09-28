import pystan
import os
import sys
import pandas as pd

from drain.step import MapResults, Call
from drain.util import read_file, dict_merge, dict_product

# default sampling args
SAMPLING_ARGS = dict(iter=500, chains=4)

def get_stan_code(filename):
    filename = os.path.join(os.path.dirname(__file__), 
                     'models', filename)
    with open(filename) as f:
        return f.read()

def extract(filename, data_args=None, model_data_args=None, sampling_args=None):
    if data_args is None:
        data_args = {}
    if data_args is None:
        sampling_args = {}
    if model_data_args is None:
        model_data_args = {}
    if sampling_args is None:
        sampling_args = {}

    sampling_args = dict_merge(SAMPLING_ARGS, sampling_args)
    model_data_args = dict_merge(MODEL_DATA_ARGS, model_data_args)

    data = BeachData(**data_args)
    data.target = True

    model_data = ModelData(**model_data_args)
    model = Call(pystan.StanModel,
                      model_code=get_stan_code(filename))
    model.target = True

    fit = FitStanModel(model=model, data=model_data), **sampling_args)
    fit.target = True

    extract = Extract(fit, model_data,
                      parameter_keys=parameter_keys)
    extract.target = True
    return extract

# dynamically create all workflows
# should replace this with evaluation in drain cmdlinea

def make_workflow(filename):
    def workflow(**model_data_args):
        return extract(filename,
            model_data_args=model_data_args,
            data_args=dict(),
        )
    return workflow

for filename in os.listdir(
        os.path.join(os.path.dirname(__file__), 'models')):
        setattr(sys.modules[__name__], 
           filename[:-5],
           make_workflow(filename))
