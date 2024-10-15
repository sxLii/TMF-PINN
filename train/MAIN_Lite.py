import sys
sys.path.append('/opt/data/private/TMF-PINN')

from utils import *
from models.TMF_PINN_Lite import SVE

# import data
## FVM
dfh = []
water_depth=[]
for i in range(1, 7):
    keys=f'resh{i}'
    file_path = f'data/dx1/resh{i}.mat'
    data = scipy.io.loadmat(file_path)
    df = pd.DataFrame(data[keys])
    dfh.append(df)
    water_depth.append(df.iloc[::100, 2:-1:2].values)

dfu = []
velocity_total=[]
for i in range(1, 7):
    keys=f'resu{i}'
    file_path = f'data/dx1/resu{i}.mat'
    data = scipy.io.loadmat(file_path)
    df = pd.DataFrame(data[keys])
    dfu.append(df)
    velocity_total.append(df.iloc[::100, 2:-1:2].values)

timeFVM=[]
for i in range(6):
    timeFVM.append(dfh[i].iloc[::100,0:1].values)
    timeFVM[i]=timeFVM[i]*60*60 # change time from 0.1h to 360s

## SWMM:
DFexp=pd.read_csv(r'data/SWMM.csv')
node_h_SWMM=DFexp.iloc[:360,15:21].values
## SWMM: time series,height and velocity
timeSWMM=DFexp.iloc[:360,0:1].values # time
time_to_seconds_vec=np.vectorize(time_to_seconds)
timeExp=time_to_seconds_vec(timeSWMM)

pipe_h_SWMM=DFexp.iloc[:360,1:7].values
pipe_u_SWMM=DFexp.iloc[:360,8:14].values
## SWMM: interpolation
node_h,pipe_h,pipe_u=[],[],[]
for i in range(6):
    node_h.append(interp1(timeFVM[i], np.hstack([timeExp, node_h_SWMM[:, i:i + 1]])))
    pipe_h.append(interp1(timeFVM[i], np.hstack([timeExp, pipe_h_SWMM[:, i:i + 1]])))
    pipe_u.append(interp1(timeFVM[i], np.hstack([timeExp, pipe_u_SWMM[:, i:i + 1]])))

# up height:1-5-6-4-4-6
index_up=[1,5,6,4,4,6]
hup=[]
for _,ins in enumerate(index_up):
    hup.append(node_h[ins-1])
# down height:
index_down=[4,2,3,6,5,5]
hdown = []
for _,ins in enumerate(index_down):
    hdown.append(node_h[ins-1])

# Topo data
# Constant
totalTime=6*60*60 #s
Length=np.array([300,300,410,500,300,310])
nm=0.0013 # S-1
# a=[1283.3,1000,1000,1000,1000,1000] # m/s
D=np.array([0.8,0.5,0.5,0.5,0.5,0.5])
Ts = 0.000001 * D
a = np.sqrt(9.81 * np.pi / 4 * D * D / Ts)
slope_=np.array([(10.7-10.4)/300,(10.0-9.5)/300,(10.1-9.3)/410,(10.4-10.1)/500,(10.4-10.0)/300,(10.1-10.0)/310])
slope=[]
for i in range(6):
    slope.append(slope_[i]*np.ones([water_depth[i].shape[1],]))


# TRAIN==========

for i in range(6):
    # 数据个数
    Nt = water_depth[i].shape[0]
    Nx = water_depth[i].shape[1]

    Nt_train = water_depth[i].shape[0]
    Nx_train = water_depth[i].shape[1]

    # EXACT:FVM
    t = np.linspace(0, totalTime, Nt)
    x = np.linspace(0, Length[i], Nx)
    u_exact = velocity_total[i][:Nt_train, :]
    h_exact = water_depth[i][:Nt_train, :]

    # imput data
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = u_exact.flatten()[:, None] 
    h_star = h_exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  ## IC @t=0s,x=1-1000m
    hh1 = h_exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))  ## upstrm BC @x=0,t=1-95s
    uu2 = u_exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))  ## dnstrm BC @x=1000,t=1-95s
    uu3 = u_exact[:, -1:]

    # BC and IC
    X_h_IC = xx1
    h_IC = hh1
    X_u_BC = np.vstack([xx2, xx3])
    X_h_BC = np.vstack([xx2, xx3])
    u_BC = np.vstack([uu2, uu3])
    h_BC = np.vstack([hup[i], hdown[i]])

    # OBS reader
    hhhobs = pipe_h[i]
    uuuobs = pipe_u[i]

    # OBS
    useObs = True
    ## obs velocity
    mid = int(np.floor(x.shape[0] / 2))
    ind_obs_u = [mid]
    t_obs_u = np.array([])
    x_obs_u = np.array([])
    u_obs = np.array([])
    for iobs in ind_obs_u:
        t_obs_u = np.append(t_obs_u, t.flatten())
        x_obs_u = np.append(x_obs_u, np.ones(Nt_train) * x[iobs])

    ind_obs_hintp = [0]
    for iobs in ind_obs_hintp:
        if np.isnan(add_noise(uuuobs[:Nt_train, 0])).any:
            u_obs = np.append(u_obs, (uuuobs[:Nt_train, iobs]))
        else:
            u_obs = np.append(u_obs, add_noise(uuuobs[:Nt_train, iobs]))

    X_u_obs = np.vstack([x_obs_u, t_obs_u]).T
    u_obs = u_obs[:, None]

    ## obs water depth
    ind_obs_h = [mid]

    t_obs_h = np.array([])
    x_obs_h = np.array([])
    h_obs = np.array([])
    for iobs in ind_obs_h:
        t_obs_h = np.append(t_obs_h, t.flatten())
        x_obs_h = np.append(x_obs_h, np.ones(Nt_train) * x[iobs])
        if np.isnan(add_noise(h_exact[:Nt_train, 0])).any:
            h_obs = np.append(h_obs, h_exact[:Nt_train, iobs])
        else:
            h_obs = np.append(h_obs, add_noise(h_exact[:Nt_train, iobs]))

    X_h_obs = np.vstack([x_obs_h, t_obs_h]).T
    h_obs = h_obs[:, None]

    # training data
    X_f_train = X_star  # Star是解析解
    Slope = np.hstack([np.array(slope[i]) for _ in range(Nt_train)])[:, None]
    Diameter=D[i]
    Cel=a[i]

    exist_mode = 0
    saved_path = 'saved_model/PINN_SVE.pickle'
    weight_path = 'saved_model/weights.out'

    layers1 = [2] + 6 * [1 * 64] + [2]
    #layers2 = [2] + 6 * [1 * 64] + [1]
    layersA = [2] + 4 * [1 * 32] + [1]
    layers2=[]
    # # Training
    model = SVE(X_h_IC,
                 X_u_BC, X_h_BC,
                 X_u_obs, X_h_obs,
                 X_f_train,
                 h_IC,
                 u_BC, h_BC,
                 u_obs, h_obs,
                 layers1, layers2, layersA,
                 lb, ub, Slope, Cel, Diameter,nm,
                 X_star, u_star, h_star,
                 ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path,
                 useObs=True)
    model.train(10000)
    #model.train_bfgs()
    #models.append(model)

    # Post Processing
    Nt_test = Nt_train  # 训练集和测试集的尺寸是一样的？
    N_test = Nt_test * Nx  ## Nt_test x Nx
    X_test = X_star[:N_test, :]
    x_test = X_test[:, 0:1]
    t_test = X_test[:, 1:2]
    u_test = u_star[:N_test, :]
    h_test = h_star[:N_test, :]
    ## predict
    u_pred2, h_pred2 = model.predict(x_test, t_test)

    ## prepare for output
    u_pred2 = u_pred2.reshape([Nt_test, Nx])
    h_pred2 = h_pred2.reshape([Nt_test, Nx])
    u_test = u_test.reshape([Nt_test, Nx])
    h_test = h_test.reshape([Nt_test, Nx])
    ## saving data
    name = f'results/PINN_lite/pipe{i+1}/'
    np.save(name + 'h_test.npy', h_test)
    np.save(name + 'h_pred2.npy', h_pred2)
    np.save(name + 'h_obs.npy', hhhobs)
    np.save(name + 'hup.npy', hup)
    np.save(name + 'hdown.npy', hdown)

    np.save(name + 'u_test.npy', u_test)
    np.save(name + 'u_pred2.npy', u_pred2)
    np.save(name + 'u_obs.npy', uuuobs)