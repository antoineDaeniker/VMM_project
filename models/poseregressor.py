import torch as th
import torch.nn as nn

from .resnet import ResnetFE
from lib.functional.softargmax import IntegralSoftargmax

# TODO replace BatchNorm with InstanceNorm !!!

class PoseRegressor(nn.Module):
    def __init__(self, 
            inplanes=2048,
            num_joints=16,
            depth_dim=1, # must be equal to Heatmap size if 3D output!
            deconv_with_bias=False,
            num_deconv_layers=3,
            num_deconv_filters=[256, 256, 256],
            num_deconv_kernels=[4, 4, 4],
            final_conv_kernel=1,
            use_instance_norm=False,
            image_size=256
            ):
        
        #### added instanceNorm option
        self.Norm2d = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        ####

        self.inplanes = inplanes
        self.deconv_with_bias = deconv_with_bias
        super(PoseRegressor, self).__init__()
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        self.final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=num_joints*depth_dim,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0
        )

        self.integral_layer = IntegralSoftargmax()
        self.image_size = image_size

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(self.Norm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        x = self.integral_layer(x)
        # now x in range [0,1]
        x = x * self.image_size
        return x

    def init_weights(self):
        print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                print('=> init {}.weight as normal(0, 0.001)'.format(name))
                print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                print('=> init {}.weight as 1'.format(name))
                print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('=> init final conv weights from normal distribution')
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print('=> init {}.weight as normal(0, 0.001)'.format(name))
                print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


class PoseNet(nn.Module): # one class for coping with both fe and pr models
    def __init__(self, **kwargs):
        super(PoseNet, self).__init__()
        params_fe = {} if 'fe' not in kwargs else kwargs['fe']
        params_pr = {} if 'pr' not in kwargs else kwargs['pr']

        self.fe = ResnetFE(**params_fe)
        self.pr = PoseRegressor(**params_pr)

    def forward(self, x):
        x = self.fe(x)
        x = self.pr(x)
        return x


# def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
#     assert isinstance(heatmaps, th.Tensor)

#     heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

#     accu_x = heatmaps.sum(dim=2)
#     accu_x = accu_x.sum(dim=2)
#     accu_y = heatmaps.sum(dim=2)
#     accu_y = accu_y.sum(dim=3)
#     accu_z = heatmaps.sum(dim=3)
#     accu_z = accu_z.sum(dim=3)

#     accu_x = accu_x * th.arange(x_dim).to(accu_x.device)
#     accu_y = accu_y * th.arange(x_dim).to(accu_y.device)
#     accu_z = accu_z * th.arange(x_dim).to(accu_z.device)

#     accu_x = accu_x.sum(dim=2, keepdim=True)
#     accu_y = accu_y.sum(dim=2, keepdim=True)
#     accu_z = accu_z.sum(dim=2, keepdim=True)

#     return accu_x, accu_y, accu_z

# def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth):
#     # global soft max
#     preds = preds.reshape((preds.shape[0], num_joints, -1))
#     preds = th.nn.functional.softmax(preds, 2)

#     # integrate heatmap into joint location
#     if output_3d:
#         x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
#     else:
#         assert 0, 'Not Implemented!' #TODO: Not Implemented

#     # now x,y,z have values from 0 to 1
#     x = x / float(hm_width)
#     y = y / float(hm_height)
#     z = z / float(hm_depth)
#     preds = th.cat((x, y, z), dim=2)
#     preds = ( preds - 0.5 ) * 2 # to make prediction values from -1 to 1
#     preds = preds.reshape((preds.shape[0], num_joints * 3))
#     return preds

# class IntegralSoftargmax(nn.Module):
#     '''
#     Given prediction tensor ("preds") of size B x (Njoints * depth_dim) x H x W 
#     does soft argmax operation, based on Integral Human Pose Regression paper.
#     Output is of a size B x Njoints x dim, where dim can be either 2 or 3.

#     In case, depth_dim == 1, implies that the input is of the "heatmap" type, 2D output.
#     Any other setting implies that model predicts the third (depth) dimension, too.

#     Softmax gives values in range from -1 to 1. 
#     To get some mm-like quantities, result must be multiplied by "joint_box_max_val" - maximum box size value.
#     '''
#     def __init__(self, num_joints, output_3d, joint_box_max_val):
#         super(IntegralSoftargmax, self).__init__()
#         self.num_joints = num_joints
#         self.output_3d = output_3d
#         self.joint_box_max_val = joint_box_max_val

#     def forward(self, preds):
#         hm_width = preds.shape[-1]
#         hm_height = preds.shape[-2]
#         hm_depth = preds.shape[-3] // self.num_joints if self.output_3d else 1

#         pred_jts = softmax_integral_tensor(preds, self.num_joints, self.output_3d, hm_width, hm_height, hm_depth)

#         pred_jts = pred_jts * self.joint_box_max_val 
#         ### predictions are from -joint_box_max_val to joint_box_max_val
#         return pred_jts





