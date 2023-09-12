# imports missing

"""
Classes for the cnn motion mixer

the final object can be initialized with the following code snippet:

motionmixer = MotionMixerCNN(input_size_spatial = 66,
               input_size_temporal = 10,
               embedding_size = 128,
               n_block = 3,
               n_output = 10,
               n_temp_layers = 2,
               n_spatial_layers = 2,
               temp_kernel_size = 5,
               spatial_kernel_size = 15)



the input must have the dimensions (batch_size, sequence_lenght, feature_lenght)
where the feature lenght is the number of joints used times the dimension for each joint (e.g. 3)

"""
import torch
import torch.nn as nn


class PoseEmbedding(nn.Module):
  """
  module to calculate embedding into a featurespace of the given pose vectors

  variables
   input_size       --- length of one pose vector (original lenght is 99, here we use e.g. 66)
   embedding_size   --- lenght of the outputvector for each pose vector (in the paper 60 was used)
  """
  def __init__(self, input_size = 66, embedding_size = 60):
      super().__init__()

      self.input_size = input_size
      self.embedding_size = embedding_size # in paper definded as C

      # for simplicity in implementation we use conv1d to simulate a mlp per pose so the input tensor does not have to be reshaped (i assume it is slower than applying a convolution)
      self.fc_W0 = conv = nn.Conv1d(self.input_size, self.embedding_size, 1, stride=1, bias = True)

  def forward(self, x):
      x = x.transpose(-1,-2)
      x = self.fc_W0(x).transpose(-1,-2)
      return x


class PosePrediction(nn.Module):
  """
  module to calculate the actual poses from featurespace for a given number of time steps.
  two linear layers are used to get from the given number of frames (e.g. 10) to the wanted number of frames (in this task also 10)
  and from the featurespace to the output space of each pose

  variables
   embedding_size   --- lenght of the outputvector for each pose vector (in the paper 60 was used)
   n_out            --- lenght of an the vector in output space (which is also the input space)
   input_len        --- number of frames/poses (in this task 10)
   output_len       --- number of wanted frames for the output (here also 10)
  """
  def __init__(self, embedding_size = 128, n_out = 66, input_len = 10, output_len = 10):  # output_len is defined as Tf in the paper
    super().__init__()

    self.embedding_size = embedding_size
    self.n_out = n_out
    self.input_len = input_len
    self.output_len = output_len

    self.fc_Wp1 = nn.Conv1d(self.input_len, self.output_len, 1, stride=1, bias = True)
    self.sigma_r = nn.ReLU()
    self.fc_Wp2 = nn.Conv1d(self.embedding_size, self.n_out, 1, stride=1, bias = True)

  def forward(self,x):

    #x = self.fc_Wp1(x)
    #x = self.sigma_r(x)
    x = self.fc_Wp2(x.transpose(-1,-2)).transpose(-1,-2)
    return x



class SEBlock(nn.Module):
  """
  definition of SE-Block according to the definition in the given paper
  """
  def __init__(self, n_frames , r=8):
    super().__init__()

    self.n_frames = n_frames
    self.r = r

    self.squeeze = nn.AdaptiveAvgPool1d(1)
    self.excite = nn.Sequential(
            nn.Linear(self.n_frames, int(self.n_frames/r), bias=False),
            nn.ReLU(),
            nn.Linear(int(self.n_frames/r), self.n_frames, bias=False),
            nn.Sigmoid()
    )

  def forward(self,x):
    y = self.squeeze(x).squeeze()
    y = self.excite(y).unsqueeze(2)
    y = y.expand(x.shape)
    z = x*y
    return z


class CNNMixer(nn.Module):
  """
  class for a general convolutional mixer block. For spatial/temporal mixer
  the input tensor needs to be transposed

  we use specifically conv2d instead of conv1d to facilitate implementation of rectangular kernels
  for conv1d is simulated with a kernel of size (1,n)

  kernel_size can be an integer or tupel of integers. integers must be odd (for now)
  """
  def __init__(self, n_features = 128, kernel_size = 3, n_layers = 2, dropout = 0.1):
    super().__init__()

    self.n_features = n_features
    self.n_layers = n_layers
    self.dropout = dropout

    if isinstance(kernel_size, int):
      self.kernel_size = (1,kernel_size)
    else:
      self.kernel_size = kernel_size

    if n_layers == 0:
      self.layers = nn.Identity()
    else:
      layer_list = []
      padding = (int(self.kernel_size[0]/2), int(self.kernel_size[1]/2))
      for i in range(self.n_layers):
        layer_list.append(nn.Conv2d(1,1,kernel_size = self.kernel_size, padding = padding))
        layer_list.append(nn.GELU())
        layer_list.append(nn.Dropout(p = self.dropout))

      self.layers = nn.Sequential(*layer_list)


  def forward(self,x):
    x = x.unsqueeze(1)
    x = self.layers(x)
    x = x.squeeze()
    return x


class TemporalMixCNN(nn.Module):
  def __init__(self, n_features = 128, kernel_size = 3, n_layers = 2):
    super().__init__()

    self.n_features = n_features
    self.n_layers = n_layers
    self.kernel_size = (1, kernel_size)

    self.cnn_block = CNNMixer(n_features = self.n_features, kernel_size = self.kernel_size, n_layers = self.n_layers)

  def forward(self,x):
    y = x.transpose(-1,-2)
    y = self.cnn_block(y)
    x = y.transpose(-1,-2)
    return x

class SpatialMixCNN(nn.Module):
  def __init__(self, n_features = 128, kernel_size = 3, n_layers = 2):
    super().__init__()

    self.n_features = n_features
    self.n_layers = n_layers
    self.kernel_size = (1, kernel_size)

    self.cnn_block = CNNMixer(n_features = self.n_features, kernel_size = self.kernel_size, n_layers = self.n_layers)

  def forward(self,x):
    x = self.cnn_block(x)
    return x


class STMixerBlock(nn.Module):
  def __init__(self, embedding_size = 128, in_len = 10, n_temp_layers = 2, temp_kernel = 3, n_spatial_layers = 2, spatial_kernel = 3):
    super().__init__()

    self.embedding_size = embedding_size
    self.in_len = in_len
    self.temp_kernel = temp_kernel
    self.spatial_kernel = spatial_kernel
    self.n_spatial_layers = n_spatial_layers
    self.n_temp_layers = n_temp_layers

    # definition of spatial mixer and first se block
    self.spatial_mixer = SpatialMixCNN(n_features = self.embedding_size, kernel_size = self.spatial_kernel, n_layers = self.n_spatial_layers)
    self.SEblock1 = SEBlock(n_frames = self.in_len, r=8)
    # definition of temporal mixer and second se block
    self.temporal_mixer = TemporalMixCNN(n_features = self.embedding_size, kernel_size = self.temp_kernel, n_layers = self.n_temp_layers)
    self.SEblock2 = SEBlock(n_frames = self.in_len, r=8)

    self.LayerNorm1 = nn.LayerNorm(self.embedding_size)
    self.LayerNorm2 = nn.LayerNorm(self.embedding_size)


  def forward(self,x):
    # first the spatial mixing block
    y = self.LayerNorm1(x)
    y = self.spatial_mixer(y)
    y = self.SEblock1(y)
    # with a skip connection
    x = x + y
    # then the temporal mixing block
    y = self.LayerNorm2(x)
    y = self.temporal_mixer(y)
    y = self.SEblock2(y)
    # with skip connection
    x = x + y

    return x

class MotionMixerCNN(nn.Module):
  def __init__(self, input_size_spatial = 66,
               input_size_temporal = 10,
               embedding_size = 128,
               n_block = 3,
               n_output = 10,
               n_temp_layers = 2,
               n_spatial_layers = 2,
               temp_kernel_size = 5,
               spatial_kernel_size = 15):
    super().__init__()

    self.input_spatial = input_size_spatial
    self.input_temporal = input_size_temporal
    self.embedding_size = embedding_size
    self.n_output = n_output
    self.n_block = n_block

    self.n_temp_layers = n_temp_layers
    self.n_spatial_layers = n_spatial_layers
    self.temp_kernel_size = temp_kernel_size
    self.spatial_kernel_size = spatial_kernel_size

    self.poseEmbedding = PoseEmbedding(input_size = self.input_spatial,
                                       embedding_size = self.embedding_size)

    if self.n_block == 0: #
      self.STMixer = nn.Identity()
    else:
      mixer_list = []
      for i in range(self.n_block):
        mixer_list.append(STMixerBlock(embedding_size = self.embedding_size,
                                       in_len = self.input_temporal,
                                       n_temp_layers = self.n_temp_layers,
                                       temp_kernel = self.temp_kernel_size,
                                       n_spatial_layers = self.n_spatial_layers,
                                       spatial_kernel = self.spatial_kernel_size))

      self.mixer = nn.Sequential(*mixer_list)

    self.posePrediction = PosePrediction(embedding_size = self.embedding_size,
                                         n_out = self.input_spatial,
                                         input_len = self.input_temporal,
                                         output_len = self.n_output)

  def forward(self,x):
    x = self.poseEmbedding(x) # embed
    x = self.mixer(x)
    x = self.posePrediction(x)

    return x