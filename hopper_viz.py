import torch
from mujoco_physics import HopperPhysics
a = torch.load('experiments/my_save')
ind = 73
truth = a["truth"]
traj = truth["observed_data"][ind][0:100:20]
true_hooper = HopperPhysics('data')
true_hooper.visualize(traj)
pred_hopper = HopperPhysics('data')
pred = a["predict"]
traj = pred[0][ind][0:100:20]
pred_hopper.visualize(traj, dirname='hopper_imgs_pred')

a = torch.load('experiments/my_save_trans')
pred_hopper = HopperPhysics('data')
pred = a["predict"]
traj = pred[0][ind][0:100:20]
pred_hopper.visualize(traj, dirname='hopper_imgs_pred_trans')