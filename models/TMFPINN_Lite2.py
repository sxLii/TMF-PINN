from typing import List, Union
import torch
from collections import OrderedDict
import torch.nn.init as init
from torch.nn.utils import clip_grad_norm_

class TMFPINNLite(torch.nn.Module):
    def __init__(
        self,
        layers: List[int],
        layers_alpha: List[int],
        lb: Union[torch.Tensor, None] = None,
        ub: Union[torch.Tensor, None] = None,
        device: torch.device = torch.device("cuda:0"),
    ) -> None:
        super(TMFPINNLite, self).__init__()
        
        # Setup for normalization bounds
        self.lb = lb.to(device) if lb is not None else None
        self.ub = ub.to(device) if ub is not None else None
        
        # Networks
        self.PINN = self._DNN(layers).to(device)
        self.AlphaDNN = self._AlphaDNN(layers_alpha).to(device)
        
        # 存储最大梯度范数的值
        self.max_grad_norm = 1.0

    def _DNN(self, layers: List[int], activation: torch.nn.Module = torch.nn.Tanh()):
        layer_list = []
        depth = len(layers) - 1
        for i in range(depth):
            linear_layer = torch.nn.Linear(layers[i], layers[i + 1])
            
            # 初始化权重和偏置
            init.xavier_uniform_(linear_layer.weight)  # Xavier 初始化
            # init.xavier_normal_(linear_layer.weight)  # Xavier 初始化
            if linear_layer.bias is not None:
                init.constant_(linear_layer.bias, 0.01)  # 偏置初始化为 0
            
            layer_list.append(("layer_%d" % i, linear_layer))
            
            # 为非最后一层添加激活函数
            if i != depth - 1:
                layer_list.append(("activation_%d" % i, activation))
        
        return torch.nn.Sequential(OrderedDict(layer_list))
    
    def _AlphaDNN(self, layers: List[int], activation: torch.nn.Module = torch.nn.Sigmoid()):
        layer_list = []
        depth = len(layers) - 1
        for i in range(depth):
            linear_layer = torch.nn.Linear(layers[i], layers[i + 1])
            
            # 初始化权重和偏置
            # init.xavier_uniform_(linear_layer.weight)
            init.xavier_normal_(linear_layer.weight)  # Xavier 初始化
            # init.constant_(linear_layer.weight, 0)  # 偏置初始化为 0, 大部分情况, 明满流都是从非满管开始的
            if linear_layer.bias is not None:
                init.constant_(linear_layer.bias, 0.01)  # 偏置初始化为 0
            
            layer_list.append(("layer_%d" % i, linear_layer))
            
            # 为非最后一层添加激活函数
            if i != depth - 1:
            # for alphaDNN, the last layer must be a linear layer, 
            # because the range of alpha is limited in [0,1]
                layer_list.append(("activation_%d" % i, activation))
        
        return torch.nn.Sequential(OrderedDict(layer_list))


    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor], tuple]) -> torch.Tensor:
        
        # max-min normalization
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        
        # AlphaDNN:[0,1]
        x_norm = (x - self.lb[0]) / (self.ub[0] - self.lb[0])
        t_norm = (t - self.lb[1]) / (self.ub[1] - self.lb[1])
        inputs_norm = torch.cat([x_norm,t_norm], dim=-1)
        
        # FourierDNN:[-1,1]
        x_scaled = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        t_scaled = 2.0 * (t - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        inputs_scaled = torch.cat([x_scaled,t_scaled], dim=-1)
        
        # forward pass
        alpha = self.AlphaDNN(inputs_norm)
        clamped_alpha = torch.clamp(alpha, min=0, max=1)
        alpha = torch.where(clamped_alpha < 1e-2, 0, clamped_alpha)
        alpha = torch.where(clamped_alpha > 1-1e-2, 1, clamped_alpha)
        
        h_and_u = self.PINN(inputs_scaled)
        
        return h_and_u, alpha
    
    # 在训练过程中进行梯度裁剪
    def clip_gradients(self):
        # 裁剪模型的梯度
        clip_grad_norm_(self.parameters(), self.max_grad_norm)