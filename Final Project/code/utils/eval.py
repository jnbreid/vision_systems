import numpy as np
import cv2
import copy
import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = torch.eye(3, 3).repeat(n, 1, 1).float().cpu() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R

def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above
    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = torch.zeros(n, 3).float().cpu()#.to(device)
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = torch.zeros(len(idx_spec1), 3).float().cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = torch.zeros(len(idx_spec2), 3).float().cpu()#.cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = torch.zeros(len(idx_remain), 3).float().cpu()#.to(device)
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul

def rotmat2euler( R ):
  """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args
    R: a 3x3 rotation matrix
  Returns
    eul: a 3x1 Euler angle representation of R
  """
  #print(R[0,2])
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul



import numpy as np
import cv2
import copy


def angle_diff(predictions, targets):
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as an np array of shape (..., n_joints)
    """
    assert predictions.shape[-1] == predictions.shape[-2] == 3
    assert targets.shape[-1] == targets.shape[-2] == 3

    ori_shape = predictions.shape[:-2]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(preds, np.transpose(targs, [0, 2, 1]))

    # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
    # the predicted and target orientations
    angles = []
    for i in range(r.shape[0]):
        aa, _ = cv2.Rodrigues(r[i].numpy())
        angles.append(np.linalg.norm(aa))
    angles = np.array(angles)

    return np.reshape(angles, ori_shape)


def euler_error(ang_pred, ang_gt):
    # only for 32 joints
    dim_full_len=ang_gt.shape[2]

    pred_expmap = ang_pred.contiguous().view(-1,dim_full_len).view(-1, 3)
    targ_expmap = ang_gt.contiguous().view(-1,dim_full_len).view(-1, 3)

    pred_eul = rotmat2euler_torch(expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = rotmat2euler_torch(expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors
"""
def euler_diff(predictions, targets):

    Computes the Euler angle error as in previous work, following
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L207
    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The Euler angle error an np array of shape (..., )

    assert predictions.shape[-1] == 3 and predictions.shape[-2] == 3
    assert targets.shape[-1] == 3 and targets.shape[-2] == 3
    n_joints = predictions.shape[-3]

    ori_shape = predictions.shape[:-2]
    
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    euler_pred_list = []
    euler_targs_list = []

    for i in range(n_joints):
      euler_pred_list.append(rotmat2euler(preds[i,:,:].view(3,3)))
      euler_targs_list.append(rotmat2euler(targs[i,:,:].view(3,3)))

    euler_preds = np.asarray(euler_pred_list) #rotmat2euler(preds)  # (N, 3)
    euler_targs = np.asarray(euler_targs_list) #rotmat2euler(targs)  # (N, 3)

    print(euler_preds.shape)
    print(euler_targs.shape)

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = np.reshape(euler_preds, [-1, n_joints*3])
    euler_targs = np.reshape(euler_targs, [-1, n_joints*3])

    print(euler_preds.shape)
    print(euler_targs.shape)

    # l2 error on euler angles
    idx_to_use = np.where(np.std(euler_targs, 0) > 1e-4)[0]
    euc_error = np.power(euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], 2)
    euc_error = np.sqrt(np.sum(euc_error, axis=1))  # (-1, ...)
    print(euc_error.shape)
    # reshape to original
    return np.reshape(euc_error, ori_shape)
"""

def eval_model(model, data_loader, criterion, device, metrik = 'angle'):
  with torch.no_grad():
      model.eval()
      
      loss_list = []

      geodesic_list = []
      pck_list = []
      eulerangle_list = []
      mpjpe_list = []
      
      for seed_frames, pred_frames in data_loader:
          seed_frames = seed_frames.to(device)
          target_frames = pred_frames.cpu()

          outputs = model(seed_frames).cpu()
                  
          loss = criterion(outputs, target_frames)
          loss_list.append(loss.item())

          if metrik == 'angle':
              # geodesic
              for i in range(target_frames.shape[0]):
                rotmatTarget = expmap2rotmat_torch(target_frames[i,:,3:].contiguous().view(-1, 3)).view(-1, int(target_frames[i,:,3:].shape[-1]/3), 3, 3).cpu()
                rotmatOut = expmap2rotmat_torch(outputs[i,:,3:].contiguous().view(-1, 3)).view(-1, outputs.shape[1], 3, 3).cpu()
                err = angle_diff(rotmatOut, rotmatTarget)
                geodesic_list.append(err)
                # eulerangle
                rotmatOut = rotmatOut.view(-1,3,3)
                rotmatTarget = rotmatTarget.view(-1,3,3)
                #for j in range(rotmatOut.shape[0]):
                #err = euler_diff(rotmatOut, rotmatTarget)#[j,:,:].view(1,3,3), rotmatTarget[j,:,:].view(1,3,3))
                #eulerangle_list.append(err)
              err = euler_error(target_frames[:,:,3:], outputs[:,:,3:])
              eulerangle_list.append(err)
              
              loss = np.mean(loss_list)

              full_geodesic = np.stack(geodesic_list)
              geodesic = full_geodesic.mean()
              full_eulerangle = np.stack(eulerangle_list)
              eulerangle = full_eulerangle.mean()

              return loss, geodesic, eulerangle
        
      return  