import torch
import numpy as np
import os
from torch.utils.data import Dataset

from utils.h3m_utils import expmap2rotmat_torch, fkl_torch, fkl_torch_fullrange, _some_variables

"""
Dataset class for the modified human3.6m dataset (the one form the slides from simon)
Data can either be loaded in exponential map format (angles between joints)
or in coord_3d format, so where each joint is represented by a 3d point in space.
Here the 3d coordinates are normalized (hip is always at position (0,0,0)) as there is a
bug in the calculation into full 3d space.
"""
class Human36(Dataset):
  def __init__(self, data_path, d_mode = 'train', data_format = 'exp_map', seed_length = 10, prediction_length = 10): # predict 10 frames from the first 10 frames (so 20 frames overall)
    self.data_path = data_path # must be the path to h3.6m directory
    self.data_format = data_format # can either be 'exp_map' or 'coord_3d'

    self.seed_length = seed_length
    self.prediction_lenght = prediction_length
    self.full_lenght = seed_length + prediction_length

    self.data_list = []

    if d_mode == 'train':
      self.data_subset = ['S1', 'S6', 'S7', 'S8', 'S9']
    elif d_mode == 'test':
      self.data_subset = ['S11']
    else:
      self.data_subset = ['S5']

    """
    function to load one full instance from the dataset given the path to the .txt file
    """
    def load_full_instance(path_to_file):

      with open(path_to_file, "r") as f:
          lines = [line.rstrip() for line in f]

      lines = [line.split(',') for line in lines]
      for i in range(len(lines)):
        lines[i] = [float(line) for line in lines[i]]
      lines = lines[::2]  # downsample from 50 to 25Hz (according to slides)
      return lines

    for direc in self.data_subset:
      individual_data_path = self.data_path + '/' + 'dataset' + '/' + direc
      scenarios = os.listdir(individual_data_path)

      for sc in scenarios:
        full_data_path = individual_data_path + '/' + sc
        instance = load_full_instance(full_data_path)
        instance = torch.tensor(instance)
        # if exponential map format is used all joints are used. this still should work for testing, however for the final models we need to restrict the vector to the relevant joints)

        if self.data_format == 'coord_3d':
          parent, offset, rotInd, expmapInd = _some_variables()

          xyz = fkl_torch(instance, parent, offset, rotInd, expmapInd)
          xyz = xyz.cpu().data.numpy()
          xyz[:,:,1:] = xyz[:,:,:0:-1]  # some changing of order for visualization

          instance = torch.tensor(xyz)

        for k in range(int(instance.shape[0]/self.full_lenght)):
          self.data_list.append(instance[k*self.full_lenght:(k+1)*self.full_lenght].clone())

    return

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    instance = self.data_list[idx]
    return instance[:self.seed_length].clone(), instance[self.seed_length:].clone()