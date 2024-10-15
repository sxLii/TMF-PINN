import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from scipy.stats import pearsonr

#plt.rcParams['font.family'] = 'Times New Roman'
def add_noise(insignal):
    """
    add noise for smooth datas
    """

    target_snr_db = 500
    # Calculate signal power and convert to dB
    sig_avg = np.mean(insignal)
    sig_avg_db = 10 * np.log10(sig_avg)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg), len(insignal))
    # Noise up the original signal
    sig_noise = insignal + noise
    return sig_noise


def time_to_seconds(time_str):
    """
    convert scheme of time form 00:01 to 60s
    """
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60

def interp1(t,ha):
    """
    return interpolation of 'ha' by the time 't'
    ha[:,0] is origin time
    ha[:,1] is the origin data, return the same length as 't'
    """
    return np.interp(t, ha[:, 0], ha[:, 1])

def h2A(h, D):
    Ts = 0.01 * D
    Afull = np.pi / 4 * D ** 2
    A = np.zeros_like(h)  # 创建与 h 同样大小的零数组，用于存储结果
    
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            h_val = h[i, j]
            if h_val >= D:
                A[i, j] = Afull + (h_val - D) * Ts
            else:
                theta = 2 * np.arccos(1 - 2 * h_val / D)
                if np.isnan(theta):
                    theta = 0.0
                A[i, j] = D ** 2 / 8 * (theta - np.sin(theta))
        
    return A

# 定义计算KGE的函数
def kge(obs, sim):
    # 将输入的多维数组展平为一维
    obs_flat = obs.flatten()
    sim_flat = sim.flatten()

    # 相关系数r
    r, _ = pearsonr(obs_flat, sim_flat)
    
    # 标准差比率alpha
    std_obs = np.std(obs_flat)
    std_sim = np.std(sim_flat)
    alpha = std_sim / std_obs
    
    # 平均值比率beta
    mean_obs = np.mean(obs_flat)
    mean_sim = np.mean(sim_flat)
    beta = mean_sim / mean_obs
    
    # 计算KGE
    kge_value = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge_value

def get_coutour_demo0(path,h_pred2,h_test,u_pred2,u_test):

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    variables = [('height $(m)$ ', h_pred2, h_test), ('velocity (m/s)', u_pred2, u_test)]

    labels = ['(a)', '(c)', '(e)', '(b)', '(d)', '(f)']

    for j, (variable, pred, test) in enumerate(variables):
        titles = [f'PREDICT:{variable}', f'Riemann Solver:{variable}', f'Error:{variable}']
        images = [pred, test, np.abs(test - pred)]

        for i, ax in enumerate(axs[:, j]):
            im = ax.imshow(images[i], aspect='auto', cmap='plasma')
            ax.set_title(titles[i], fontsize=30)
            cbar = plt.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=18)  #

            if i == 2:
                ax.set_xlabel('Distance (m) ', fontsize=18)
            if j == 0:
                ax.set_ylabel('Time (h) ', fontsize=18)

            # ax.set_xticks([0, 50, 100, 150, 200, 250], [0, 100,200,300,400,500], fontsize=18)
            # ax.set_yticks([0, 100, 200, 300,400,500,600], [0, 1, 2, 3,4,5,6], fontsize=18)
            ax.text(0.05, 0.05, labels[i + j * 3], transform=ax.transAxes, fontsize=22, weight='bold', color='white')

    plt.tight_layout()
    plt.savefig(path, dpi=500, bbox_inches='tight')  # 会裁掉多余的白边