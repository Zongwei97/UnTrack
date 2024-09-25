import torch
import torch.nn as nn
import torch.nn.functional as F

# class Expert(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(Expert, self).__init__()
#         self.layer = nn.Linear(in_features, out_features)
#
#     def forward(self, x):
#         return  F.relu(self.layer(x))
#
#
# class Router(nn.Module):
#     def __init__(self, in_features, num_experts):
#         super(Router, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(in_features, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_experts)
#         )
#
#     def forward(self, x):
#         return F.softmax(self.network(x), dim=1)
#
#
# # class MoE(nn.Module):
# #     def __init(self, in_features, out_features, num_experts):
# #         super(MoE, self).__init__()
# #         self.gate = nn.Linear(in_features, num_experts)
# #         self.experts = nn.ModuleList([Expert(in_features, out_features) for i in range(num_experts)])
# #
# #     def foward(self, x):
# #         gate_output = F.softmax(self.gate(x), dim=1)
# #         expert_outputs = [expert(x) for  expert in self.experts]
# #         final_output = 0
# #         for i, expert_output in enumerate(expert_outputs):
# #             final_output += gate_output[:, i].unsqueeze(1) * expert_output
# #         return final_output
#
#
# class MoE(nn.Module):
#     def __init(self, in_features, out_features, num_experts):
#         super(MoE, self).__init__()
#         self.gate = Router(in_features, num_experts)
#         self.experts = nn.ModuleList([Expert(in_features, out_features) for i in range(num_experts)])
#
#     def foward(self, x):
#         gate_output = self.gate(x)
#         scores, indices = torch.sort(gate_output, dim=1, descending=True)
#         final_output = 0
#         for i in range(6):
#             expert_output = self.experts[indices[:, i]](x)
#             final_output += scores[:, i].unsqueeze(1) * expert_output
#         return final_output
class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(768,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,768),
        )

    def forward(self, x):
        p_task = F.softmax(self.layer(x), dim=-1)
        return self.layer(x), p_task


class Router(nn.Module):
    def __init__(self, num_experts):
        super(Router, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.network = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool(x)
        #print('x.pool',x.shape)
        x = x.squeeze(-1)
        #print('squeeze',x.shape)
        return F.softmax(self.network(x), dim=1)


class MoE(nn.Module):
    def __init__(self, num_experts=2):
        super(MoE, self).__init__()
        self.gate = Router(num_experts)
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])

    def forward(self, x):
        #print('inputshape',x.shape) (32,64,768)
        gate_output = self.gate(x) #(32,64,10)
        #print('gateoutput',gate_output.shape) (32,10)
        topk_values, topk_indices = torch.topk(gate_output, 2, dim=1)
        #print('topk_indices',topk_indices.shape)(32,6)
        #print('ttttttt', topk_indices.t().shape)# (6,32)
        #print('topk_value',topk_values.shape)(32,6)
        #x_reshape = x.reshape(-1, x.shape[-1])
        #print('x_reshape', x_reshape.shape)(2048,768)
        #topk_indices - topk_indices.permute(1,0)
        expert_output = torch.stack([expert(x)[-1] for expert in self.experts])
        #print('expert_output', expert_output.shape)#(10,32,64,768)
        #print('unsqueeze', topk_indices.unsqueeze(2).expand(-1, -1, expert_output.shape[-1]).shape) #(32,6,768)
        topk_indices = topk_indices.t().unsqueeze(2)
        topk_indices =topk_indices.unsqueeze(3)
        #print('unsqueeze2',topk_indices.shape)
        expert_output_topk = expert_output.gather(0, topk_indices.expand(-1, -1, expert_output.shape[-2],expert_output.shape[-1]))
        #print('expertoutputtopk',expert_output_topk.shape)#6,32,64,768
        # expert_outputs = [self.experts[idx](x) for idx in topk_indices]
        final_output = 0
        for i, expert_output in enumerate(expert_output_topk):
            weight = topk_values[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(0)
            #print(weight.shape)
            final_output += weight*expert_output
            #print(final_output.shape)
            #print('use MoE')
        final_output = final_output.squeeze(0)
        return final_output

    def mutual_info_loss(self, p_z, p_t_given_z):
        H_z = -torch.sum(p_z * torch.log(p_z), dim=1)
        H_t_given_z = -torch.sum(p_t_given_z * torch.log(p_t_given_z), dim=1)
        L_mi = H_z - torch.sum(p_z * H_t_given_z, dim=1)

        return L_mi


if __name__ == "__main__":
    model = MoE().cuda()
    inp = torch.randn(32, 64, 768).cuda()
    out = model(inp)
