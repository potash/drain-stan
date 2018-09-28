from drain.step import Step
import joblib
import os


class FitStanModel(Step):
    def __init__(self, model, data, **kwargs):
        Step.__init__(self, model=model, inputs=[model, data], kwargs=kwargs)

    def run(self, model, data):
        return model.sampling(data=data, **self.kwargs)
        
    def dump(self):
        self.setup_dump()
        joblib.dump(self.model.result, os.path.join(self._dump_dirname, 'model.pkl'))
        joblib.dump(self.result, os.path.join(self._dump_dirname, 'result.pkl'))

    def load(self):
        model = joblib.load(
                os.path.join(self._output_dirname, 'dump', 'model.pkl'))
        self.result = joblib.load(
                os.path.join(self._output_dirname, 'dump', 'result.pkl'))
