import torch
from torch import nn
from torch.nn import functional as F
from airfoil_DDPM import Unet,airfoil_diffusion,airfoil_diffusion_multitask
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.tensorboard import SummaryWriter
#from mlp_data import MyData
import pandas as pd
import csv
import numpy as np
import arifoil_DDPM_tools
import matplotlib.pyplot as plt
import subprocess
import os

def partial_load_state_dict(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['models']
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    return model

def read_xfoil_polar_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        reading_data = False
        for line in lines:
            if "alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr" in line:
                reading_data = True
                continue
            if reading_data and not line.startswith('#') and line.strip():
                values = line.split()
                try:
                    data.append([float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4])])
                except ValueError:
                    pass
    return np.array(data)

def run_xfoil_for_angles(airfoil_filename, aoa_start,aoa_end,aoa_step):
    reynolds_number=7e6
    mach_number=0.75
    xfoil_path = r'D:/Xfoil/xfoil.exe'
    current_directory = os.getcwd()
    command = f'cd {current_directory} && {xfoil_path} < xfoil_commands.txt'
    os.remove('D:/Xfoil/airfoil_polars.txt')

    with open('xfoil_commands.txt', 'w') as fid:
        fid.write(f'LOAD {airfoil_filename}\n')
        fid.write('OPER\n')
        fid.write('ITER 20\n')
        fid.write(f'VISC {reynolds_number}\n')
        fid.write(f'MACH {mach_number}\n')
        #fid.write('PPAR\nN 150\n\n')
        fid.write(f'ALFA {aoa_start}\n')
        fid.write('PACC\n')
        fid.write(f'{'D:\\Xfoil\\airfoil_polars.txt'}\n\n')
        fid.write(f'ASEQ {aoa_start} {aoa_end} {aoa_step}\n')
        fid.write('PACC\n')
        fid.write('\nQUIT\n')

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        status = process.wait()
        if status!= 0:
            return None
    except Exception as e:
        print(f'XFOIL 执行出现错误：{e}')
        return None

def read_polar_files(file_path):
    values=read_xfoil_polar_file(file_path)
    results = {'cl': [], 'cd': [], 'cm': []}
    results['cl'].append(values[:,1])
    results['cd'].append(values[:,2])
    results['cm'].append(values[:,4])
    os.remove('D:/Xfoil/airfoil_polars.txt')
    return results

    
def dat_file_gen(X,Y_up,Y_down, output_file_name):
    X = np.concatenate([X, X[-1:0:-1]])
    y = np.concatenate([Y_up, Y_down[-1:0:-1]])
    output = np.column_stack([X, y])
    with open(output_file_name, 'w') as fileID:
        N = len(X)
        fileID.write(f'{N}\n')
        for i in range(output.shape[0]):
            row = '\t'.join(map(str, output[i, :]))
            fileID.write(f'	     {row}\n')

df_model=Unet(1, 20,context_size_1=3,context_size_2=3, down_sampling_dim=2, dropout = 0.).cuda()
checkpoint = torch.load('...pth', map_location=torch.device('cuda'),weights_only=True)   ### 加载神经网络模型
df_model = partial_load_state_dict(df_model, checkpoint)

#af=airfoil_diffusion(df_model)
af=airfoil_diffusion_multitask(df_model)

#x=torch.tensor([[[1,1,4,5,1,4,1,9,1,9,8,1,0,1,9,1,9,8,1,0]]])
#x = x.to(torch.float32).cuda()
#mean = x.mean()
#std = x.std()
#normalized_x = (x - mean) / std
#x=af(x, context=None, CFG=6, t_max=1000)
#context=torch.tensor([0.013258,0.002096,0.005272,0.069566,0.161616,2.432779,0.013258,0.002096,0.005272,0.069566,0.17,2.0]).reshape(-1,1,6).cuda()
#context=torch.tensor([0.013258,0.002096,0.005272,0.069566,0.161616,2.432779]).reshape(-1,1,6).cuda()


context_1=np.array([[0.008496,0.1211,0.379]])#前缘半径、最大厚度、最大厚度所在位置、上表面最高点处曲率半径
context_2=np.array([[0.1405,0.00386,-0.0952],[0.3253,0.00411,-0.0962],[0.4536,0.00637,-0.0863],[0.6133,0.00773,-0.0795],[0.7931,0.00862,-0.0717]])#任意数量的Cl，Cd，Cm  NACA0012
context_trimmed = np.array(context_1)
context_2_t = np.array(context_2)

context_trimmed[:, 0] = (context_1[:, 0] - 0.0063)/0.054
context_trimmed[:, 1] = (context_1[:, 1] - 0.09)/0.14
context_trimmed[:, 2] = (context_1[:, 2] - 0.33)/0.64
context_2_t[:, 0] = (context_2[:, 0] - 0)*10
context_2_t[:, 1] = np.sqrt(context_2[:, 1]) * 10
context_2_t[:, 2] = (context_2[:, 2] - 0)*10

#context_trimmed[:, 3] = np.log(context[:, 3])

context_trimmed=torch.tensor(context_trimmed,dtype=torch.float32).reshape(-1,1,3).cuda()
context_2_t=torch.tensor(context_2_t,dtype=torch.float32).reshape(-1,1,3).cuda()
#print(context_trimmed)
#originx=torch.tensor([0.138931,0.213787,0.041699,0.483469,-0.099642,0.41611,-0.010288,0.346384,0.135582,0.289172,-0.113609,-0.029857,-0.032536,0.01981,-0.001328,0.062312,0.016896,0.062442,0.062345,0.048032,0.138931,0.213787,0.041699,0.483469,-0.099642,0.41611,-0.010288,0.346384,0.135582,0.289172,-0.113609,-0.029857,-0.032536,0.01981,-0.001328,0.062312,0.016896,0.062442,0.062345,0.048032]).reshape(-1,1,20).cuda()
#time=torch.tensor([[500.0],[500.0]]).cuda()
#x=af(originx,context=context,CFG=2)
#print(x)
#z=af(originx,context=context,CFG=1)
#print(z)
k=af(None,context_1=context_trimmed,context_2=context_2_t,CFG=1.4,t_max=400)##################
print(k[0][0])
#l=af(None,context=None,CFG=1)
#print(l)
#R,t,t_co,up_R=airfoil_diffusion_model_tools.CST_cal(k[0][0])
geolabel=arifoil_DDPM_tools.geo_label_cal(k[:][0])
R,t,t_co=geolabel[0][0],geolabel[0][1],geolabel[0][2]
print(f'Target_R:{context_1[0][0]:.4f},   Target_thickness:{context_1[0][1]:.3f},   Target_thickness_coordinate:{context_1[0][2]:.3f}')
print(f'Generate_R:{R:.4f}, Generate_thickness:{t:.3f}, Generate_thickness_coordinate:{t_co:.3f}')

###计算气动特性
n = 75
num = np.linspace(-1, 1, n)
X_cal = 0.5 - 0.5 * np.sin(num / 2 * np.pi)
X_cal[:20] = np.linspace(1, X_cal[20], 20)
Y_up_cal,Y_down_cal=arifoil_DDPM_tools.locate_plot(k[0][0],X=X_cal)

dat_file_gen(X_cal,Y_up_cal,Y_down_cal, 'D:/Xfoil/airfoil.dat')
angles_list = [-1, 0, 1, 2, 3]
run_xfoil_for_angles('D:/Xfoil/airfoil.dat', -3,4,0.2)
#print(results)

