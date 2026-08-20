from typing import List, Union
import torch
from collections import OrderedDict
import torch.nn.init as init
from torch.nn.utils import clip_grad_norm_

class TMFPINN(torch.nn.Module):
    def __init__(
        self,
        layers: List[int],
        layers_alpha: List[int],
        lb: Union[torch.Tensor, None] = None,
        ub: Union[torch.Tensor, None] = None,
        device: torch.device = torch.device("cuda:0"),
    ) -> None:
        super(TMFPINN, self).__init__()
        
        # Setup for normalization bounds
        self.lb = lb.to(device) if lb is not None else None
        self.ub = ub.to(device) if ub is not None else None
        
        # Networks
        self.layers = layers
        self.AlphaDNN = self.DNN(layers_alpha).to(device)

        # Initialize multi-scale Fourier features
        self.W1_t = torch.nn.Parameter(torch.randn(1, layers[0] // 2, device=device) * 1.0, requires_grad=True)
        self.W2_t = torch.nn.Parameter(torch.randn(1, layers[0] // 2, device=device) * 10, requires_grad=True)
        self.W1_x = torch.nn.Parameter(torch.randn(1, layers[0] // 2, device=device) * 1.0, requires_grad=True)

        # Initialize weights and biases
        self.weights, self.biases = self.initialize_WandB(layers, device)

        # 存储最大梯度范数的值
        self.max_grad_norm = 1.0

    def xavier_init(self, size: List[int], device: torch.device) -> torch.nn.Parameter:
        in_dim, out_dim = size
        std = (2.0 / (in_dim + out_dim)) ** 0.5
        return torch.nn.Parameter(torch.empty(in_dim, out_dim, device=device).normal_(0, std))

    def initialize_WandB(self, layers: List[int], device: torch.device):
        weights = torch.nn.ParameterList()
        biases = torch.nn.ParameterList()

        for i in range(len(layers) - 2):
            W = self.xavier_init([layers[i], layers[i + 1]], device)
            b = torch.nn.Parameter(0.01*torch.ones(1, layers[i + 1], device=device))
            weights.append(W)
            biases.append(b)

        W_final = self.xavier_init([2 * layers[-2], layers[-1]], device)
        b_final = torch.nn.Parameter(0.01*torch.ones(1, layers[-1], device=device))
        weights.append(W_final)
        biases.append(b_final)

        return weights, biases
    
    def DNN(self, layers: List[int], activation: torch.nn.Module = torch.nn.Sigmoid()):
        layer_list = []
        depth = len(layers) - 1
        for i in range(depth):
            linear_layer = torch.nn.Linear(layers[i], layers[i + 1])
            
            # 初始化权重和偏置
            # init.xavier_uniform_(linear_layer.weight)  # Xavier 初始化
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

    def FourierDNN(self, X: torch.Tensor) -> torch.Tensor:
        # Fourier feature encodings
        x = X[:, 0:1]
        t = X[:, 1:2]

        H1_t = torch.cat([torch.sin(torch.matmul(t, self.W1_t)), torch.cos(torch.matmul(t, self.W1_t))], dim=1)
        H2_t = torch.cat([torch.sin(torch.matmul(t, self.W2_t)), torch.cos(torch.matmul(t, self.W2_t))], dim=1)
        H1_x = torch.cat([torch.matmul(x, self.W1_x), torch.matmul(x, self.W1_x)], dim=1)

        # Neural network forward pass
        for l in range(len(self.weights) - 1):
            W = self.weights[l]
            b = self.biases[l]
            H1_t = torch.tanh(torch.matmul(H1_t, W) + b)
            H2_t = torch.tanh(torch.matmul(H2_t, W) + b)
            H1_x = torch.tanh(torch.matmul(H1_x, W) + b)

        # Merge the outputs by concatenation
        H1 = H1_t * H1_x
        H2 = H2_t * H1_x
        H = torch.cat([H1, H2], dim=1)

        W_final = self.weights[-1]
        b_final = self.biases[-1]
        return torch.matmul(H, W_final) + b_final
    
    # 在训练过程中进行梯度裁剪
    def clip_gradients(self):
        # 裁剪模型的梯度
        clip_grad_norm_(self.parameters(), self.max_grad_norm)


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
        
        h_and_u = self.FourierDNN(inputs_scaled)
        
        return h_and_u, alpha
    
# # Example usage
# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # Placeholder values for demonstration
#     layers = [2, 50, 50, 2]
#     layers_alpha = [2, 50, 50, 1]

#     model = TMFPINN(layers, layers_alpha).to(device)

#     x = torch.rand(100, 1).to(device)
#     t = torch.rand(100, 1).to(device)
    
#     inputs = (x, t)
#     # 如果 inputs 是元组，转换为 Tensor
#     if isinstance(inputs, tuple):
#         inputs = [inp if isinstance(inp, torch.Tensor) else torch.tensor(inp) for inp in inputs]

#     # 将处理后的张量拼接起来
#     inputs = torch.cat(inputs, dim=-1)
    
#     h_and_q, alpha = model(inputs)
#     print("Output h_and_q shape:", h_and_q.shape)
#     print("Output alpha shape:", alpha.shape)