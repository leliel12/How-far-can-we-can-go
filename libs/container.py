
import glob
import os

import numpy as np

import pandas as pd

class Container(dict):
    
    def __dir__(self):
        return list(self.keys())
    
    def __repr__(self):
        resume = str({
            k: len(v) if isinstance(v, pd.DataFrame) else type(v) 
            for k, v in sorted(self.items())})
        return "<Container({})>".format(resume)
    
    def __getattr__(self, an):
        if an not in self.keys():
            return super(Container, self).__getattr__(an)
        return self[an]
    
    def __setattr__(self, an, av):
        self[an] = av
        
    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass
    
    
        
def read(path):
    data = {}
    for fpath in glob.glob(os.path.join(path, "*.npy")):
        fname = os.path.basename(fpath).split("_", 1)[0]
        key = os.path.splitext(fname)[0]
        print("Loading '{}'...".format(fpath))
        data[key] = pd.DataFrame(np.load(fpath))
    return Container(data)
        
    