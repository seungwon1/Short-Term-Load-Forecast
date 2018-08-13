
# coding: utf-8
import numpy as np
import tensorflow as tf
from util_func import optim
from util_func.optim import *
from util_func.classifiers.load_forecast import *
from matplotlib import pyplot as plt


class solver(object):
    
    def __init__(self, model, data, **kwargs):
        
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
             
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.iteration = kwargs.pop('iteration', 50000)
        self.batch_size = kwargs.pop('batch_size', 5000)
        self.print_every = kwargs.pop('print_every', 1000)
        self.verbose = kwargs.pop('verbose', True)

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 0.0)
        self.num_epochs = kwargs.pop('num_epochs', 30000)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()        
        

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = [100, 100, 100]
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d        
   
    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, num_train)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config        
            
            
    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
            
    def check_accuracy(self, X, y):
        """
        Check accuracy of the model on the provided data.
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        Returns:
        - acc: mean absolute error, root mean square error, maen absolute percentage error
        """
        scores = self.model.loss(X)
        mae = np.mean(np.abs(y-scores))
        rmse = np.sqrt(np.mean(np.square(y-scores)))
        mape = np.mean((np.abs(y - scores))/(y+1e-3))*100
        acc = [mae, rmse, mape]

        return acc
        
    def train_prev(self):
                
        loss_history = []
        acc_history = []
        
        for i in range(self.iteration):
            
            loss, grads = self.model.loss(self.X_train, self.y_train)
            mae = self.check_mae(mode = 'training')
            loss_history.append(loss)
            acc_history.append(mae)
            
            if i % self.print_every == 0:
                if self.verbose == True:
                    print('iteration = %s, loss = %s, mae = %s'%(i, loss, mae))
                    
            for key, arr in self.model.param.items():
                self.model.param[key] = self.model.param[key] - (self.learning_rate)*grads[key]
                
        
    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // num_train, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))
                

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] += self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                    )
                val_acc = self.check_accuracy(self.X_val, self.y_val,
                    )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    if self.epoch % num_iterations == 0:
                        print('(Epoch(Iteration) %d / %d)' % (self.epoch, self.num_epochs))
                        print('train acc, val_acc  = ' , train_acc, val_acc)
                          
                # Keep track of the best model
                if val_acc[0] <= self.best_val_acc[0] and val_acc[1] <= self.best_val_acc[1] :
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
        
                        
            
            
            
            
            
            
            