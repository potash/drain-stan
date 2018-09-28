from drain.step import Step, MapResults
import pandas as pd
import logging

class Extract(Step):
    """
    Extracts samples from a StanFit object into a pandas DataFrame
    and optionally renames the parameter indexes to match the original data.
    """
    def __init__(self, fit, indexes, pars=None, parameter_keys=None):
        """
        Args:
            fit: Step producing a StanFit object
            data: a dictionary of key: index pairs, e.g. {'mu': [1150, 2251,...]}
            parameter_keys: optional dictionary mapping parameter names
                names to index keys, e.g. {'mu':'address_id'}
        """
        if parameter_keys is None:
            parameter_keys = {}
        Step.__init__(self, inputs=[fit, indexes],
                      pars=pars,
                      parameter_keys=parameter_keys)

    def run(self, fit, indexes):
        pars = fit.model_pars
        if self.pars is not None:
            pars = set(self.pars).intersection(fit.model_pars)

            if len(pars) < len(self.pars):
                logging.warning('Parameters do not exist in model: %s' % set(self.pars).difference(fit.model_pars))

        extract = fit.extract(pars=pars)
        constructor = {1:pd.Series, 2:pd.DataFrame, 3:pd.Panel}
        for p in extract:
            if extract[p].ndim < 4:
                extract[p] = constructor[extract[p].ndim](extract[p])

        for p, k in self.parameter_keys.items():
            if p in extract:
                if isinstance(extract[p], pd.DataFrame):
                    extract[p].columns = indexes[k]
                    extract[p].columns.name = k
                elif isinstance(extract[p], pd.Panel):
                    if isinstance(k, tuple) and len(k) == 2:
                        for i in range(2):
                            extract[p].set_axis(axis=i+1,labels=indexes[k[i]])
                            extract[p].axes[i+1].name = k[i]
                    else:
                        logging.warning("Provide two axes for panel labels: %s" % p)

            else:
                logging.warning('Parameter not found: %s' % p)

        return extract
