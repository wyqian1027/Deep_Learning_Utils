import torch
import time
import numpy as np
import matplotlib.pyplot as plt


class Timer:
    ''' simple Timer '''
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.init_time = time.time()
        self.times = [0.0]
    
    def stop(self, show_interval=False):
        self.times.append(time.time()-self.init_time)
        return self.times[-1] - self.times[-2] if show_interval else self.times[-1]
    
    def get_avg_interval(self, stopfirst=False):
        if stopfirst: self.stop()
        assert len(self.times)>=2
        return self.times[-1] / (len(self.times)-1)

class Counter:
    ''' Counter for multiple variables
    '''
    def __init__(self, *keys, keep_all=True):
        self.keys = keys
        self.keep_all = keep_all
        self.reset()

    def add(self, *values):
        assert len(values) == len(self.keys)
        for k, v in zip(self.keys, values):
            self.update_func(k, v)

    def get(self, key=None):
        if key == None: return self.counts
        return self.counts.get(key)

    def reset(self):
        if self.keep_all:
            self.counts = {k: [] for k in self.keys}
            def f(k, v): self.counts[k].append(v)
            self.update_func = f
        else:
            self.counts = {k: 0 for k in self.keys}
            def f(k, v): self.counts[k] += v
            self.update_func = f
    
    def plot(self, *keys):
        for k in keys:
            assert k in self.keys
            plt.plot(history.get(k), label=k)
        plt.legend()
        plt.show()


class StateDictManager:
    ''' Manage state dict in epochs'''
    def __init__(self, model, filename, best_acc=0):
	    assert '.' not in filename
        self.model = model
        self.filename = filename
        self.best_acc = best_acc

    def get_name(self, epoch):
        return f"{self.filename}_epoch{epoch}.state_dict"
    
    def save_dict(self, cur_acc, epoch):
        if cur_acc > self.best_acc:
            torch.save(self.model.state_dict(), self.get_name(epoch))
            self.best_acc = cur_acc
            print(f'model state_dict saved to {self.filename}.')

    def load_dict(self, epoch):
        state_dict = torch.load(self.get_name(epoch))
        self.model.load_state_dict(state_dict)
        self.model.eval()



def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_params(net):
    return sum(param.numel() for name, param in net.named_parameters())



    
