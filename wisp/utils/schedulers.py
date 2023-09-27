import math
import numpy as np

class DecayScheduler(object):
    '''A simple class for decaying schedule of various hyperparameters.'''

    def __init__(self, total_steps, decay_name='fix', start=0, end=0, params=None):
        self.decay_name = decay_name
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.params = params

    def __call__(self, step):
        if self.decay_name == 'fix':
            return self.start
        elif self.decay_name == 'linear':
            if step>self.total_steps:
                return self.end
            return self.start + (self.end - self.start) * step / self.total_steps
        elif self.decay_name == 'exp':
            return max(self.end, self.start*(np.exp(-np.log(1/self.params['temperature'])*step/self.total_steps/self.params['decay_period'])))
            # return self.start * (self.end / self.start) ** (step / self.total_steps)
        elif self.decay_name == 'inv_sqrt':
            return self.start * (self.total_steps / (self.total_steps + step)) ** 0.5
        elif self.decay_name == 'cosine':
            return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(step / self.total_steps * math.pi))
        else:
            raise ValueError('Unknown decay name: {}'.format(self.decay_name))