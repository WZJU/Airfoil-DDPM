###本代码用于输入9阶CST参数，输出其坐标点、前缘半径、最大厚度、最大厚度位置、上表面最高点半径
import math
import sympy as sp
import scipy.special
from scipy.optimize import minimize_scalar
import numpy as np
import os
from scipy.spatial.distance import directed_hausdorff
import subprocess
from scipy.spatial import distance_matrix

# 定义符号变量
x = sp.Symbol('x')
# 定义函数
def cst_model_up(c):
    return x**0.5 * (1 - x)**1 * sum(c[i] * scipy.special.comb(9, i) * x**i * (1 - x)**(9 - i) for i in range(10))

def cst_model_down(c):
    return x**0.5 * (1 - x)**1 * sum(c[i+10] * scipy.special.comb(9, i) * x**i * (1 - x)**(9 - i) for i in range(10))

def thick(c):
    expr=x**0.5 * (1 - x)**1 * sum(c[i] * scipy.special.comb(9, i) * x**i * (1 - x)**(9 - i)-c[i+10] * scipy.special.comb(9, i) * x**i * (1 - x)**(9 - i) for i in range(10))
    return sp.lambdify(x, expr)

def R(c,x):
    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9=c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]
    R=sp.Abs(((0.5*(x - 1)*(-1.0*c0*(x - 1)**9 + 9.0*c1*x*(x - 1)**8 - 36.0*c2*x**2*(x - 1)**7 + 84.0*c3*x**3*(x - 1)**6 - 126.0*c4*x**4*(x - 1)**5 + 126.0*c5*x**5*(x - 1)**4 - 84.0*c6*x**6*(x - 1)**3 + 36.0*c7*x**7*(x - 1)**2 - 9.0*c8*x**8*(x - 1) + 1.0*c9*x**9)/x**0.5 + x**0.5*(x - 1)*(-9.0*c0*(x - 1)**8 + 72.0*c1*x*(x - 1)**7 + 9.0*c1*(x - 1)**8 - 252.0*c2*x**2*(x - 1)**6 - 72.0*c2*x*(x - 1)**7 + 504.0*c3*x**3*(x - 1)**5 + 252.0*c3*x**2*(x - 1)**6 - 630.0*c4*x**4*(x - 1)**4 - 504.0*c4*x**3*(x - 1)**5 + 504.0*c5*x**5*(x - 1)**3 + 630.0*c5*x**4*(x - 1)**4 - 252.0*c6*x**6*(x - 1)**2 - 504.0*c6*x**5*(x - 1)**3 + 36.0*c7*x**7*(2*x - 2) + 252.0*c7*x**6*(x - 1)**2 - 9.0*c8*x**8 - 72.0*c8*x**7*(x - 1) + 9.0*c9*x**8) + x**0.5*(-1.0*c0*(x - 1)**9 + 9.0*c1*x*(x - 1)**8 - 36.0*c2*x**2*(x - 1)**7 + 84.0*c3*x**3*(x - 1)**6 - 126.0*c4*x**4*(x - 1)**5 + 126.0*c5*x**5*(x - 1)**4 - 84.0*c6*x**6*(x - 1)**3 + 36.0*c7*x**7*(x - 1)**2 - 9.0*c8*x**8*(x - 1) + 1.0*c9*x**9))**2 + 1)**1.5/(-0.25*(x - 1)*(-1.0*c0*(x - 1)**9 + 9.0*c1*x*(x - 1)**8 - 36.0*c2*x**2*(x - 1)**7 + 84.0*c3*x**3*(x - 1)**6 - 126.0*c4*x**4*(x - 1)**5 + 126.0*c5*x**5*(x - 1)**4 - 84.0*c6*x**6*(x - 1)**3 + 36.0*c7*x**7*(x - 1)**2 - 9.0*c8*x**8*(x - 1) + 1.0*c9*x**9)/x**1.5 + 1.0*(x - 1)*(-9.0*c0*(x - 1)**8 + 72.0*c1*x*(x - 1)**7 + 9.0*c1*(x - 1)**8 - 252.0*c2*x**2*(x - 1)**6 - 72.0*c2*x*(x - 1)**7 + 504.0*c3*x**3*(x - 1)**5 + 252.0*c3*x**2*(x - 1)**6 - 630.0*c4*x**4*(x - 1)**4 - 504.0*c4*x**3*(x - 1)**5 + 504.0*c5*x**5*(x - 1)**3 + 630.0*c5*x**4*(x - 1)**4 - 252.0*c6*x**6*(x - 1)**2 - 504.0*c6*x**5*(x - 1)**3 + 36.0*c7*x**7*(2*x - 2) + 252.0*c7*x**6*(x - 1)**2 - 9.0*c8*x**8 - 72.0*c8*x**7*(x - 1) + 9.0*c9*x**8)/x**0.5 + 1.0*(-1.0*c0*(x - 1)**9 + 9.0*c1*x*(x - 1)**8 - 36.0*c2*x**2*(x - 1)**7 + 84.0*c3*x**3*(x - 1)**6 - 126.0*c4*x**4*(x - 1)**5 + 126.0*c5*x**5*(x - 1)**4 - 84.0*c6*x**6*(x - 1)**3 + 36.0*c7*x**7*(x - 1)**2 - 9.0*c8*x**8*(x - 1) + 1.0*c9*x**9)/x**0.5 + x**0.5*(x - 1)*(-72.0*c0*(x - 1)**7 + 504.0*c1*x*(x - 1)**6 + 144.0*c1*(x - 1)**7 - 1512.0*c2*x**2*(x - 1)**5 - 1008.0*c2*x*(x - 1)**6 - 72.0*c2*(x - 1)**7 + 2520.0*c3*x**3*(x - 1)**4 + 3024.0*c3*x**2*(x - 1)**5 + 504.0*c3*x*(x - 1)**6 - 2520.0*c4*x**4*(x - 1)**3 - 5040.0*c4*x**3*(x - 1)**4 - 1512.0*c4*x**2*(x - 1)**5 + 1512.0*c5*x**5*(x - 1)**2 + 5040.0*c5*x**4*(x - 1)**3 + 2520.0*c5*x**3*(x - 1)**4 - 252.0*c6*x**6*(2*x - 2) - 3024.0*c6*x**5*(x - 1)**2 - 2520.0*c6*x**4*(x - 1)**3 + 72.0*c7*x**7 + 504.0*c7*x**6*(2*x - 2) + 1512.0*c7*x**5*(x - 1)**2 - 144.0*c8*x**7 - 504.0*c8*x**6*(x - 1) + 72.0*c9*x**7) + 2*x**0.5*(-9.0*c0*(x - 1)**8 + 72.0*c1*x*(x - 1)**7 + 9.0*c1*(x - 1)**8 - 252.0*c2*x**2*(x - 1)**6 - 72.0*c2*x*(x - 1)**7 + 504.0*c3*x**3*(x - 1)**5 + 252.0*c3*x**2*(x - 1)**6 - 630.0*c4*x**4*(x - 1)**4 - 504.0*c4*x**3*(x - 1)**5 + 504.0*c5*x**5*(x - 1)**3 + 630.0*c5*x**4*(x - 1)**4 - 252.0*c6*x**6*(x - 1)**2 - 504.0*c6*x**5*(x - 1)**3 + 36.0*c7*x**7*(2*x - 2) + 252.0*c7*x**6*(x - 1)**2 - 9.0*c8*x**8 - 72.0*c8*x**7*(x - 1) + 9.0*c9*x**8)))
    return R

def geo_label_cal(CST):
    #输入的c是一个batch的数据，size是[n,20]
    #首先是计算其前缘半径，这里可以写死x，把c作为变量
    #还是只能用循环的形式
    geo_label = []
    for i in range(CST.shape[0]):
        local_CST=CST[i]
        Ra=math.sqrt(R(local_CST,0.00001)*R(local_CST[10:],0.00001))#前缘半径
        thickness=thick(local_CST)
        result = minimize_scalar(lambda x: -thickness(x), bounds=(0.1, 0.7), method='bounded')
        maximum_thick = thickness(result.x)
        maximum_position = result.x
        geo_label.append([Ra,maximum_thick,maximum_position])
    return geo_label

def locate_plot(CST,X=None):
    if X is None:
        X=X = np.linspace(0, 1, 200)
    up_surface = cst_model_up(CST)
    down_surface=cst_model_down(CST)
    Y_up=[up_surface.subs(x, value) for value in X]
    Y_down=[down_surface.subs(x, value) for value in X]
    return Y_up,Y_down

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

def read_polar_files(file_path):
    values=read_xfoil_polar_file(file_path)
    results = {'cl': [], 'cd': [], 'cm': []}
    if np.any(values):
        results['cl'].append(values[:,1])
        results['cd'].append(values[:,2])
        results['cm'].append(values[:,4])
    #os.remove(file_path)
    return results

def Polar_cal(CST):
    ###计算气动特性
    n = 75
    num = np.linspace(-1, 1, n)
    X_cal = 0.5 - 0.5 * np.sin(num / 2 * np.pi)
    X_cal[:20] = np.linspace(1, X_cal[20], 20)
    Y_up_cal,Y_down_cal=locate_plot(CST,X=X_cal)
    dat_file_gen(X_cal,Y_up_cal,Y_down_cal, 'D:/Xfoil/airfoil.dat')
    run_xfoil_for_angles('D:/Xfoil/airfoil.dat', -3,4,0.2)
    results=read_polar_files('D:/Xfoil/airfoil_polars.txt')
    return results

def distance_cal(Target,Polar):
    d1, d2, pairs = directed_hausdorff(Target, Polar)#第一个值是第一个点集到第二个点集的单向豪斯多夫距离；第二个值是第二个点集到第一个点集的单向豪斯多夫距离；第三个值是两个点集中对应点对的索引。
    return d1#根据需求，返回第一个值

def average_nearest_neighbor_distance(Target, Polar):
    if not np.any(Polar):
        return 10
    dist_matrix = distance_matrix(Target, Polar)
    nearest_distances_1_to_2 = [min(row) for row in dist_matrix]
    avg_dist_1_to_2 = sum(nearest_distances_1_to_2) / len(Target)
    return avg_dist_1_to_2