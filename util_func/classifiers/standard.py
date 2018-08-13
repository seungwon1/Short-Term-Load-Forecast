# coding: utf-8
import numpy as np
import tensorflow as tf
from util_func.layers import *

class std_demand(object):
    """ compute standard demand of peak load and wether for a day """
    
    
    def __init__(self, data, , **kwargs):
        
        self.month = 
        
    
    
    date = day - 1
    diff = np.abs((month - month[date])[:, 1]) + 2.5*np.abs((month - month[date])[:, 2]) # give extra weight to the difference of max temperature
    diff_sort = np.argsort(diff)
    mark = 0
    idx_list = []
    if month[date, 3] == 6 or month[date, 3] == 7:
        pass
        
    # collect indicies of similar days
    
    for i in diff_sort:
        if i >= date:
            continue
    
        else:
            if month[i, 3] == 6 or month[i, 3] == 7:
                pass
            elif date - i <= 15:    # references should be made within two weeks
                idx_list.append(i)
                mark += 1
                if mark == 3:       # refer to three days for computing standard demand
                    break
    
    # compute standard load demand 
    idx_list = np.array(idx_list)
    diff2 = date - idx_list
    diff_rate = (1+np.amax(diff2)) - diff2
    diff_rate = diff_rate / np.sum(diff_rate, axis = 0)
    std_load = diff_rate[0]*month[idx_list[0], 7:31] + diff_rate[1]*month[idx_list[1], 7:31] + diff_rate[2]*month[idx_list[2], 7:31]
    
    # compute standard temperature, humadity demand
    diff3 = np.abs(month[idx_list, 1] - month[date, 1])  + 2.5*np.abs(month[idx_list, 2] - month[date, 2])
    diff4 = np.sum(diff3) - diff3
    diff_rate2 = diff4 / np.sum(diff4)
    std_temp = diff_rate2[0]*month[idx_list[0],[1,2,4,0]] + diff_rate2[1]*month[idx_list[1],[1,2,4,0]] + diff_rate2[2]*month[idx_list[2],[1,2,4,0]]
    std_temp[3] = month[date, 0]
    
    return std_load, std_temp