import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
import torch.nn.functional as F
from math import sqrt
import csv
from einops import rearrange, repeat
import numpy as np

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

#Resnet
class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
#downsampling and up sampling
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)
    
class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()  #Sigmoid
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)
    
class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
    
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

#CrossAttention
class CrossAttention(nn.Module):
    def __init__(self, query_size, context_size=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_size = dim_head * heads
        context_size = default(context_size, query_size)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q1 = nn.Linear(query_size, 40, bias=False)
        self.to_q2 = nn.Linear(40, inner_size, bias=False)
        self.to_k1 = nn.Linear(query_size, 40, bias=False)
        self.act_k = nn.SiLU()
        self.to_k2 = nn.Linear(40, inner_size, bias=False)
        self.to_v1 = nn.Linear(query_size, 40, bias=False)
        self.to_v2 = nn.Linear(40, inner_size, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_size, query_size),
            nn.Dropout(dropout)
        )
        if context_size is not None:
            self.c_to_k1 = nn.Linear(context_size,40)
            self.c_to_k2 = nn.Linear(40,inner_size)
            self.act = nn.SiLU()


    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q1(x)
        q = self.to_q2(q)
        v = self.to_v1(x)
        v = self.to_v2(v)
        if context is not None:
            context=context.repeat(1, x.shape[1], 1)
            k = self.c_to_k1(context)
            k = self.act(k)
            k = self.c_to_k2(k)
        else:
            k = self.to_k1(x)
            k = self.act_k(k)
            k = self.to_k2(k)
            

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b (h n) d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale #softmax(qk/sqrt(d))v

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, 'b (h n) d -> b n (h d)', h=h)
        return self.to_out(out)

class UNet_Block_down(nn.Module):
    def __init__(self, dim, query_size, context_size, down_sampling_dim, time_emb_dim=None, dropout = 0.):
        super().__init__()
        self.resnet=ResnetBlock(dim, dim, time_emb_dim = time_emb_dim, dropout = dropout)
        self.crossattention=CrossAttention(query_size=query_size, context_size=context_size, heads=8, dim_head=64, dropout=dropout)
        self.downsample=Downsample(dim, dim_out = down_sampling_dim)

    def forward(self, x,time_emb = None, context=None,):
        x=self.resnet(x,time_emb)
        x=self.crossattention(x, context)
        x=self.downsample(x)
        return x

class UNet_Block_up(nn.Module):
    def __init__(self, down_sampling_dim, query_size, context_size, up_sampling_dim, time_emb_dim=None, dropout = 0.):
        super().__init__()
        self.resnet=ResnetBlock(down_sampling_dim, down_sampling_dim, time_emb_dim = time_emb_dim, dropout = dropout)
        self.crossattention=CrossAttention(query_size=query_size, context_size=context_size, heads=8, dim_head=64, dropout=dropout)
        self.downsample=Upsample(down_sampling_dim, dim_out = up_sampling_dim)

    def forward(self, x,time_emb = None, context=None):
        x=self.resnet(x,time_emb)
        x=self.crossattention(x, context)
        x=self.downsample(x)
        return x

class Unet(nn.Module):
    def __init__(self, input_dim, query_size,context_size_1,context_size_2, down_sampling_dim, dropout = 0.):
        super().__init__()
        down_sampling_dim_1=down_sampling_dim
        down_sampling_dim_2=down_sampling_dim*2
        self.block_down_1=UNet_Block_down( input_dim, query_size, context_size_1, down_sampling_dim_1, time_emb_dim=1, dropout = dropout)
        self.block_down_2=UNet_Block_down( down_sampling_dim_1, query_size//2, context_size_1, down_sampling_dim_2, time_emb_dim=1, dropout = dropout)

        self.block_up_2=UNet_Block_up( down_sampling_dim_2, query_size//4, context_size_2, down_sampling_dim_1, time_emb_dim=1, dropout = dropout)
        self.block_up_1=UNet_Block_up( down_sampling_dim_1, query_size//2, context_size_2, input_dim, time_emb_dim=1, dropout = dropout)

    def forward(self, x, time_emb, context_1=None, context_2=None):
        d1=self.block_down_1(x, time_emb, context_1)
        d2=self.block_down_2(d1,time_emb, context_1)
        u2=self.block_up_2(d2,context=context_2)
        u1=self.block_up_1(u2+d1,context=context_2)
        return u1


def cal_alpha_bar(alpha, max_t):
    alphabar=torch.empty(max_t)
    temp=1
    for i in range(max_t):
        temp=temp*alpha[i]
        alphabar[i]=temp
    return alphabar



class airfoil_diffusion(nn.Module):
    #def __init__(self, input_dim, query_size,context_size, down_sampling_dim, model,dropout = 0):
    def __init__(self, model):
        super().__init__()
        #self.model=Unet(input_dim, query_size,context_size, down_sampling_dim, dropout = 0.)
        
        #checkpoint = torch.load(model_filename, map_location=torch.device('cuda'),weights_only=True)   ### 加载神经网络模型
        #self.model.load_state_dict(checkpoint['models'])
        self.model=model
        on_cuda = next(self.model.parameters()).is_cuda

        self.beta=torch.linspace(0.0001, 0.02, steps=1000)
        self.alpha=1-self.beta
        self.alphabar=cal_alpha_bar(self.alpha, 1000)
        
        if on_cuda:
            self.beta = self.beta.cuda()
            self.alpha = self.alpha.cuda()
            self.alphabar = self.alphabar.cuda()
            self.model = self.model.cuda()



    def forward(self, x, context_1=None, context_2=None, CFG=6, t_max=1000):
        mean = 0.0  
        std = 1.0  
        if x is None and context_1 is None and context_2 is None:
            x = torch.normal(mean=mean, std=std, size=(20,)).reshape(1,-1,20).cuda()
        elif x is None and context_1 is not None:
            x = torch.normal(mean=mean, std=std, size=(20*context_1.shape[0],)).reshape(context_1.shape[0],-1,20).cuda()
        elif x is None and context_2 is not None:
            x = torch.normal(mean=mean, std=std, size=(20*context_2.shape[0],)).reshape(context_2.shape[0],-1,20).cuda()
        else:
            xmean = x.mean()
            xstd = x.std()
            x = (x - xmean) / xstd*2
        
        if context_1 is not None or context_2 is not None:
            for t in range(t_max-1, 0, -1):
                time_embedding = torch.tensor(t, dtype=torch.float32).unsqueeze(0).expand(x.shape[0], -1).cuda()
                if context_1 is not None:
                    context_1= context_1.cuda()
                if context_2 is not None:
                    context_2= context_2.cuda()
                hidden_without_context=self.model(x, time_embedding, context_1=None, context_2=None)
                hidden_with_context=self.model(x, time_embedding, context_1=context_1, context_2=context_2)
                noise_pred= hidden_without_context + CFG*( hidden_with_context-hidden_without_context)
                add_noise=torch.normal(mean=mean, std=std, size=(20,)).reshape(1,-1,20).cuda()
                x=1/sqrt(self.alpha[t])*(x-(1-self.alpha[t])*noise_pred/sqrt(1-self.alphabar[t]))+self.beta[t]*(1-self.alphabar[t-1])/(1-self.alphabar[t])*add_noise
                #with open('.../generate.csv', 'a', newline='') as csvfile:
                    #writer = csv.writer(csvfile)
                    #writer.writerow([t, x])
                    #print(f't={t}')
        else:
            for t in range(t_max-1, 0, -1):
                time_embedding = torch.tensor(t, dtype=torch.float32).unsqueeze(0).expand(x.shape[0], -1).cuda()
                noise_pred=self.model(x, time_embedding)
                add_noise=torch.normal(mean=mean, std=std, size=(20,)).reshape(1,-1,20).cuda()
                x=1/sqrt(self.alpha[t])*(x-(1-self.alpha[t])*noise_pred/sqrt(1-self.alphabar[t]))+self.beta[t]*(1-self.alphabar[t-1])/(1-self.alphabar[t])*add_noise
                #x=(x-(1-self.alpha[t])*noise_pred/sqrt(1-self.alphabar[t]))+self.beta[t]*(1-self.alphabar[t-1])/(1-self.alphabar[t])*add_noise
                #with open('.../generate.csv', 'a', newline='') as csvfile:
                    #writer = csv.writer(csvfile)
                    #x_list = x.cpu().tolist()
                    #sqrtalpha=1/sqrt(self.alpha[t])
                    #coeff_2=(1-self.alpha[t])/sqrt(1-self.alphabar[t])
                    #coeff_3=self.beta[t]*(1-self.alphabar[t-1])/(1-self.alphabar[t])
                    #written_list = [t] + x_list[0]#+[sqrtalpha]+[coeff_2.item()]+[coeff_3.item()]
                    #written_list = [t] + x_list[0]
                    #writer.writerow(written_list)
                    #print(f't={t}')
        return x

def tensor_combine(tensor1,tensor2):
    combinations1 = []
    combinations2 = []
    if (tensor1 is None)or(tensor2 is None):
        return tensor1,tensor2
    for i in range(tensor1.shape[0]):
        for j in range(tensor2.shape[0]):
            combinations1.append(tensor1[i])
            combinations2.append(tensor2[j])

    return torch.stack(combinations1), torch.stack(combinations2)


def weighted_average(tensor):
    n = tensor.shape[0]
    weights = torch.zeros(n)
    for i in range(n):
        distances = torch.tensor([torch.sum((tensor[i] - tensor[j])**2) for j in range(n)])
        weights[i] = torch.exp(-distances[i])
    weights = weights / torch.sum(weights)
    weights=weights.cuda()
    output=torch.sum(tensor * weights.view(n, 1, 1), dim=0)
    return output.reshape(-1,output.shape[-2],output.shape[-1])


class airfoil_diffusion_multitask(nn.Module):
    #def __init__(self, input_dim, query_size,context_size, down_sampling_dim, model,dropout = 0):
    def __init__(self, model):
        super().__init__()
        #self.model=Unet(input_dim, query_size,context_size, down_sampling_dim, dropout = 0.)
        #checkpoint = torch.load(model_filename, map_location=torch.device('cuda'),weights_only=True)   ### 加载神经网络模型
        #self.model.load_state_dict(checkpoint['models'])
        self.model=model
        on_cuda = next(self.model.parameters()).is_cuda

        self.beta=torch.linspace(0.0001, 0.02, steps=1000)
        self.alpha=1-self.beta
        self.alphabar=cal_alpha_bar(self.alpha, 1000)
        if on_cuda:
            self.beta = self.beta.cuda()
            self.alpha = self.alpha.cuda()
            self.alphabar = self.alphabar.cuda()
            self.model = self.model.cuda()

    def forward(self, x, context_1, context_2, CFG=6, t_max=1000):
        mean = 0.0  
        std = 1.0  
        if x is None:
            x = torch.normal(mean=mean, std=std, size=(20,)).reshape(1,-1,20).cuda()
            if context_1 is not None:
                 shape_1=context_1.shape[0]
            else:
                shape_1=1
            if context_2 is not None:
                 shape_2=context_2.shape[0]
            else:
                shape_2=1

            x.repeat(shape_1*shape_2, 1, 1).cuda()
        else:
            xmean = x.mean()
            xstd = x.std()
            x = (x - xmean) / xstd*2
            x=x.repeat(context_1.shape[0]*context_2.shape[0], 1, 1)
        context_1_combined,context_2_combined=tensor_combine(context_1,context_2)
        for t in range(t_max-1, 0, -1):
            time_embedding = torch.tensor(t, dtype=torch.float32).unsqueeze(0).expand(x.shape[0], -1).cuda()
            hidden_without_context=self.model(x, time_embedding, context_1=None, context_2=None)
            hidden_with_context=self.model(x, time_embedding, context_1=context_1_combined, context_2=context_2_combined)
            hidden_with_context=weighted_average(hidden_with_context)
            noise_pred= hidden_without_context + CFG*( hidden_with_context-hidden_without_context)
            add_noise=torch.normal(mean=mean, std=std, size=(20,)).reshape(1,-1,20).cuda()
            x=1/sqrt(self.alpha[t])*(x-(1-self.alpha[t])*noise_pred/sqrt(1-self.alphabar[t]))+self.beta[t]*(1-self.alphabar[t-1])/(1-self.alphabar[t])*add_noise
        return x

