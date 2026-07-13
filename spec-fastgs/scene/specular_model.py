import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.spec_utils import SpecularNetwork, SpecularNetworkReal
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, get_linear_noise_func


class SpecularModel:
    def __init__(
        self,
        asg_degree=24,
        is_real=False,
        is_indoor=False,
        asg_num_theta=-1,
        asg_num_phi=-1,
        specular_hidden=-1,
        specular_layers=-1,
    ):
        if is_real:
            self.specular = SpecularNetworkReal(
                asg_degree,
                is_indoor,
                asg_num_theta,
                asg_num_phi,
                specular_hidden,
                specular_layers,
            ).cuda()
        else:
            self.specular = SpecularNetwork(
                asg_degree,
                asg_num_theta,
                asg_num_phi,
                specular_hidden,
                specular_layers,
            ).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, asg_feature, viewdir, normal):
        return self.specular(asg_feature, viewdir, normal)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.specular.parameters()),
             'lr': training_args.feature_lr / 10,
             "name": "specular"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.specular_scheduler_args = get_linear_noise_func(lr_init=training_args.feature_lr,
                                                             lr_final=training_args.feature_lr / 20,
                                                             lr_delay_mult=training_args.position_lr_delay_mult,
                                                             max_steps=training_args.specular_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "specular/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.specular.state_dict(), os.path.join(out_weights_path, 'specular.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "specular"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "specular/iteration_{}/specular.pth".format(loaded_iter))
        self.specular.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "specular":
                lr = self.specular_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
