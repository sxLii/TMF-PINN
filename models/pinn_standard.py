#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import tensorflow as tf

print(tf.__version__)

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
tf.set_random_seed(1234)

class SVE:
    
    DTYPE=tf.float32
    # Initialize the class
    def __init__(self, X_h_IC, 
                      X_u_BC, X_h_BC,
                      X_u_obs, X_h_obs,
                      X_f, 
                      h_IC,
                      u_BC, h_BC,
                      u_obs,h_obs, 
                      layers,
                      lb, ub, S, a, D,nm,
                      X_star, u_star, h_star,
                      lr=5e-4,
                      ExistModel=0, uhDir='', wDir='', useObs=True):

        # Count for callback function
        self.count=0
        self.nm = nm

        self.lb = lb
        self.ub = ub
        self.S = S  ## channel slope
        self.a = a
        self.D = D
        self.useObs = useObs

        # test data
        self.X_star = X_star
        self.u_star = u_star
        self.h_star = h_star

        
        self.x_h_IC = X_h_IC[:,0:1]
        self.t_h_IC = X_h_IC[:,1:2]

        self.x_u_BC = X_u_BC[:,0:1]
        self.t_u_BC = X_u_BC[:,1:2]
        self.x_h_BC = X_h_BC[:,0:1]
        self.t_h_BC = X_h_BC[:,1:2]

        self.x_u_obs = X_u_obs[:,0:1]
        self.t_u_obs = X_u_obs[:,1:2]
        self.x_h_obs = X_h_obs[:,0:1]
        self.t_h_obs = X_h_obs[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.h_IC = h_IC
        self.u_BC = u_BC
        self.h_BC = h_BC
        self.u_obs = u_obs
        self.h_obs = h_obs
        
        # layers
        self.layers = layers

        # initialize NN
        self.weights1, self.biases1 = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data on velocities (inside the domain)
        self.x_u_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_u_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])
        self.x_h_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_h_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])

        self.x_h_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_h_IC.shape[1]])
        self.t_h_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_h_IC.shape[1]])
        self.h_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_IC.shape[1]])

        self.x_u_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_u_BC.shape[1]])
        self.t_u_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_u_BC.shape[1]])
        self.u_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.u_BC.shape[1]])
        self.x_h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_h_BC.shape[1]])
        self.t_h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_h_BC.shape[1]])
        self.h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_BC.shape[1]])

        self.x_u_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_u_obs.shape[1]])
        self.t_u_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_u_obs.shape[1]])
        self.u_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.u_obs.shape[1]])
        self.x_h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_h_obs.shape[1]])
        self.t_h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_h_obs.shape[1]])
        self.h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_obs.shape[1]])
        
        
        self.x_f_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])  

        # physics informed neural networks
        self.u_pred, self.h_pred = self.net_uh(self.x_u_tf, self.t_u_tf)
        self.u_IC_pred, self.h_IC_pred = self.net_uh(self.x_h_IC_tf, self.t_h_IC_tf)
        self.u_BC_pred, self.h_BC_pred = self.net_uh(self.x_u_BC_tf, self.t_u_BC_tf)
        if self.useObs:
            self.u_obs_pred, self.h_obs_pred = self.net_uh(self.x_u_obs_tf, self.t_u_obs_tf)
        self.eq1_pred, self.eq2_pred = self.net_f(self.x_f_tf, self.t_f_tf) 
                                    
        # loss
        self.loss_f_c  = tf.reduce_mean(tf.square(self.eq1_pred)) ## continuity
        self.loss_f_m  = tf.reduce_mean(tf.square(self.eq2_pred)) ## momentum
        self.loss_f    = self.loss_f_c + self.loss_f_m

        self.loss_BC_u = tf.reduce_mean(tf.square(self.u_BC_tf - self.u_BC_pred))
        self.loss_BC_h = tf.reduce_mean(tf.square(self.h_BC_tf - self.h_BC_pred))
        self.loss_BCs = self.loss_BC_u + self.loss_BC_h

        self.loss_IC_h = tf.reduce_mean(tf.square(self.h_IC_tf - self.h_IC_pred))
        self.loss_ICs = self.loss_IC_h

        self.loss = self.loss_f + self.loss_BCs + self.loss_ICs

        if self.useObs:
            self.loss_obs_u = tf.reduce_mean(tf.square(self.u_obs_tf - self.u_obs_pred)) 
            self.loss_obs_h = tf.reduce_mean(tf.square(self.h_obs_tf - self.h_obs_pred))
            self.loss_obs = self.loss_obs_u + self.loss_obs_h
            self.loss += self.loss_obs
        

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0e-10,
                                                                           'gtol' : 0.000001})

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step,
                                                       5000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, global_step=self.global_step)

        ## Loss logger
        self.loss_f_c_log = []
        self.loss_f_m_log = []
        self.loss_BC_u_log  = []
        self.loss_BC_h_log  = []
        self.loss_IC_h_log  = []
        self.loss_obs_u_log = []
        self.loss_obs_h_log = []
        self.l2_u_error_log = []
        self.l2_h_error_log = []

        init = tf.global_variables_initializer()
        
        self.sess.run(init)


                               
        
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=self.DTYPE), dtype=self.DTYPE)
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=self.DTYPE)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_alpha(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            if l==0: # fls
                H= tf.sin(tf.add(tf.matmul(H, W), b))
            else:
                H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        #Y = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        Y = tf.add(tf.matmul(H, W), b)

        return Y



    def net_uh(self, x, t):
        X = 2.0*(tf.concat([x, t], 1) - self.lb)/(self.ub - self.lb) - 1.0
        hu = self.neural_net(X, self.weights1, self.biases1)

        h = hu[:,0:1]
        u = hu[:, 1:2]

        return u, h
         
    def net_f(self, x_f, t_f):
        X_f = 2.0*(tf.concat([x_f, t_f], 1) - self.lb)/(self.ub - self.lb) - 1.0
        hu = self.neural_net(X_f, self.weights1, self.biases1)

        h = hu[:,0:1]
        u = hu[:, 1:2]

        u_t = tf.gradients(u, t_f)[0]
        u_x = tf.gradients(u, x_f)[0]
        
        h_t = tf.gradients(h, t_f)[0]
        h_x = tf.gradients(h, x_f)[0]
        
        eq1 = self.fun_r_mass(u, h, h_t, h_x, u_x)
        eq2 = self.fun_r_momentum(u, h, u_t, u_x, h_x)
           
        return eq1, eq2

    def fun_r_mass(self, u, h, h_t, h_x, u_x):

        ht = tf.clip_by_value(h, clip_value_min=1e-4, clip_value_max=self.D - 1e12)
        theta = 2 * tf.acos(1 - 2 * ht / self.D)
        theta = tf.clip_by_value(theta, clip_value_min=1e-3, clip_value_max=1.96 * np.pi)
        A_B = self.D / 8 * (theta - tf.sin(theta)) / tf.cos(theta / 4)

        # return h_t + u * h_x + alpha * self.a * self.a / 9.81 * u_x + (1 - alpha) * h * u_x
        return h_t + u * h_x + A_B * u_x

    def fun_r_momentum(self, u, h, u_t, u_x, h_x):
        n = self.nm

        ht = tf.clip_by_value(h, clip_value_min=1e-4, clip_value_max=self.D - 1e12)
        theta = 2 * tf.acos(1 - 2 * ht / self.D)
        theta = tf.clip_by_value(theta, clip_value_min=1e-3, clip_value_max=1.96 * np.pi)
        # A_B = self.D / 8 * (theta - tf.sin(theta)) / tf.cos(theta / 4)
        R = self.D / 4 * (1 - tf.sin(theta) / theta)

        return (u_t + u * u_x + 9.81 * h_x + 9.81 * (n * n * tf.abs(u) * u / tf.pow(tf.square(R), 2. / 3) - self.S))

    def callback_obs(self, loss, loss_f_c, loss_f_m, loss_BC_u, loss_BC_h, loss_IC_h, loss_obs_u, loss_obs_h):
        self.count = self.count+1
        print('{} th iterations, Loss: {:.3e}, Loss_f_c: {:.3e}, Loss_f_m: {:.3e}'.format(self.count, loss, loss_f_c, loss_f_m))
        self.loss_f_c_log.append(loss_f_c)
        self.loss_f_m_log.append(loss_f_m)
        self.loss_BC_u_log.append(loss_BC_u)
        self.loss_BC_h_log.append(loss_BC_h)
        self.loss_IC_h_log.append(loss_IC_h)
        self.loss_obs_u_log.append(loss_obs_u)
        self.loss_obs_h_log.append(loss_obs_h)

    def callback(self, loss, loss_f_c, loss_f_m, loss_BC_u, loss_BC_h, loss_IC_h):
        self.count = self.count+1
        print('{} th iterations, Loss: {:.3e}, Loss_f_c: {:.3e}, Loss_f_m: {:.3e}'.format(self.count, loss, loss_f_c, loss_f_m))
        self.loss_f_c_log.append(loss_f_c)
        self.loss_f_m_log.append(loss_f_m)
        self.loss_BC_u_log.append(loss_BC_u)
        self.loss_BC_h_log.append(loss_BC_h)
        self.loss_IC_h_log.append(loss_IC_h)

    def train(self, num_epochs):
        
        tf_dict = {self.x_h_IC_tf: self.x_h_IC, self.t_h_IC_tf: self.t_h_IC, self.h_IC_tf: self.h_IC, 
                   self.x_u_BC_tf: self.x_u_BC, self.t_u_BC_tf: self.t_u_BC, self.u_BC_tf: self.u_BC, 
                   self.x_h_BC_tf: self.x_h_BC, self.t_h_BC_tf: self.t_h_BC, self.h_BC_tf: self.h_BC,
                   self.x_u_obs_tf: self.x_u_obs, self.t_u_obs_tf: self.t_u_obs, self.u_obs_tf: self.u_obs,
                   self.x_h_obs_tf: self.x_h_obs, self.t_h_obs_tf: self.t_h_obs, self.h_obs_tf: self.h_obs,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                   }
        
        for it in range(num_epochs):
            
            start_time = time.time()
            self.sess.run(self.train_op_Adam, tf_dict)
                
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                learning_rate = self.sess.run(self.learning_rate)
                print('It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                         %(it, loss_value, elapsed, learning_rate))

                u_pred, h_pred = self.predict(self.X_star[:,0:1], self.X_star[:,1:2])
                error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                error_h = np.linalg.norm(self.h_star - h_pred, 2) / np.linalg.norm(self.h_star, 2)

                if self.useObs: 
                    loss_BC_u, loss_BC_h, loss_IC_h, loss_obs_u, loss_obs_h, loss_f_c, loss_f_m = \
                        self.sess.run([self.loss_BC_u, self.loss_BC_h, self.loss_IC_h, self.loss_obs_u, self.loss_obs_h, self.loss_f_c, self.loss_f_m], tf_dict)
                    print ('Loss_BC_u: %.3e, Loss_BC_h: %.3e, Loss_IC_h: %.3e, Loss_obs_u: %.3e, Loss_obs_h: %.3e, Loss_f_c: %.3e, Loss_f_m: %.3e, Error u: %.3e, Error h: %.3e'
                            %(loss_BC_u, loss_BC_h, loss_IC_h, loss_obs_u, loss_obs_h, loss_f_c, loss_f_m, error_u, error_h))
                else:
                    loss_BC_u, loss_BC_h, loss_IC_h, loss_f_c, loss_f_m = \
                        self.sess.run([self.loss_BC_u, self.loss_BC_h, self.loss_IC_h, self.loss_f_c, self.loss_f_m], tf_dict)
                    print ('Loss_BC_u: %.3e, Loss_BC_h: %.3e, Loss_IC_h: %.3e, Loss_f_c: %.3e, Loss_f_m: %.3e, Error u: %.3e, Error h: %.3e'
                            %(loss_BC_u, loss_BC_h, loss_IC_h, loss_f_c, loss_f_m, error_u, error_h))
                
                self.loss_f_c_log.append(loss_f_c)
                self.loss_f_m_log.append(loss_f_m)
                self.loss_BC_u_log.append(loss_BC_u)
                self.loss_BC_h_log.append(loss_BC_h)
                self.loss_IC_h_log.append(loss_IC_h)
                if self.useObs:
                    self.loss_obs_u_log.append(loss_obs_u)
                    self.loss_obs_h_log.append(loss_obs_h)
                self.l2_u_error_log.append(error_u)
                self.l2_h_error_log.append(error_h)

    def train_bfgs(self):

        tf_dict = {self.x_h_IC_tf: self.x_h_IC, self.t_h_IC_tf: self.t_h_IC, self.h_IC_tf: self.h_IC,
                   self.x_u_BC_tf: self.x_u_BC, self.t_u_BC_tf: self.t_u_BC, self.u_BC_tf: self.u_BC,
                   self.x_h_BC_tf: self.x_h_BC, self.t_h_BC_tf: self.t_h_BC, self.h_BC_tf: self.h_BC,
                   self.x_u_obs_tf: self.x_u_obs, self.t_u_obs_tf: self.t_u_obs, self.u_obs_tf: self.u_obs,
                   self.x_h_obs_tf: self.x_h_obs, self.t_h_obs_tf: self.t_h_obs, self.h_obs_tf: self.h_obs,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                   }

        if self.useObs:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_f_c, self.loss_f_m, self.loss_BC_u, self.loss_BC_h, self.loss_IC_h, self.loss_obs_u, self.loss_obs_h],
                                    loss_callback=self.callback_obs)
        else:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_f_c, self.loss_f_m, self.loss_BC_u, self.loss_BC_h, self.loss_IC_h],
                                    loss_callback=self.callback)


    def predict(self, x_star, t_star):
        
        tf_dict = {self.x_u_tf: x_star, self.t_u_tf: t_star,
                   self.x_h_tf: x_star, self.t_h_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        h_star = self.sess.run(self.h_pred, tf_dict)
        
        return u_star, h_star
