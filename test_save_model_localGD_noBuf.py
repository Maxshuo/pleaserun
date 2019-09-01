from mec_env_var import *
from helper import *
import tensorflow as tf
import ipdb as pdb
import time

for i in range(10):

    print('---------' + str(i) + '------------')

    MAX_EPISODE = 1000
    MAX_EPISODE_LEN = 200

    NUM_T = 1
    NUM_R = 4
    SIGMA2 = 1e-9

    t_factor = 0.9
    noise_sigma = 0.12

    config = {'state_dim': 3, 'action_dim': 2};
    train_config = {'minibatch_size': 64, 'actor_lr': 0.0001, 'tau': 0.001,
                    'critic_lr': 0.001, 'gamma': 0.99, 'buffer_size': 250000,
                    'random_seed': int(time.clock() * 1000 % 1000), 'noise_sigma': noise_sigma, 'sigma2': SIGMA2}
    user_config = [{'id': '1', 'model': 'AR', 'num_r': NUM_R, 'rate': 3.0, 'dis': 100, 'action_bound': 2,
                    'data_buf_size': 100, 't_factor': t_factor, 'penalty': 1000}]

    print(user_config)

    # 1. include all user in the system according to the user_config
    user_list = []
    for info in user_config:
        info.update(config)
        user_list.append(MecTermGD(info, train_config, 'local'))
        print('Initialization OK!----> user ' + info['id'])

    # 2. create the simulation env
    env = MecSvrEnv(user_list, NUM_R, SIGMA2, MAX_EPISODE_LEN)

    # #Create a saver object which will save all the variables
    # saver = tf.train.Saver()

    res_r = []
    res_p = []
    res_b = []
    res_o = []
    res_d = []
    # 3. start to explore for each episode
    for i in range(MAX_EPISODE):

        cur_init_ds_ep = env.reset()

        cur_r_ep = np.zeros(len(user_list))
        cur_p_ep = np.zeros(len(user_list))
        cur_op_ep = np.zeros(len(user_list))
        cur_ts_ep = np.zeros(len(user_list))
        cur_ps_ep = np.zeros(len(user_list))
        cur_rs_ep = np.zeros(len(user_list))
        cur_ds_ep = np.zeros(len(user_list))
        cur_ch_ep = np.zeros(len(user_list))
        cur_of_ep = np.zeros(len(user_list))

        for j in range(MAX_EPISODE_LEN):

            # first try to transmit from current state
            [cur_r, done, cur_p, cur_op, temp, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch, cur_of] = env.step_transmit()

            cur_r_ep += cur_r
            cur_p_ep += cur_p
            cur_op_ep += cur_op
            cur_ts_ep += cur_ts
            cur_ps_ep += cur_ps
            cur_rs_ep += cur_rs
            cur_ds_ep += cur_ds
            cur_ch_ep += cur_ch
            cur_of_ep += cur_of

            if done:
                res_r.append(cur_r_ep / MAX_EPISODE_LEN)
                res_p.append(cur_p_ep / MAX_EPISODE_LEN)
                res_b.append(cur_ds_ep / MAX_EPISODE_LEN)
                res_o.append(cur_of_ep / MAX_EPISODE_LEN)
                res_d.append(cur_ds)
                print('%d:r:%s,p:%s,op:%s,tr:%s,pr:%s,rev:%s,dbuf:%s,ch:%s,ibuf:%s,rbuf:%s' % (
                i, cur_r_ep / MAX_EPISODE_LEN, cur_p_ep / MAX_EPISODE_LEN, cur_op_ep / MAX_EPISODE_LEN,
                cur_ts_ep / MAX_EPISODE_LEN, cur_ps_ep / MAX_EPISODE_LEN, cur_rs_ep / MAX_EPISODE_LEN,
                cur_ds_ep / MAX_EPISODE_LEN, cur_ch_ep / MAX_EPISODE_LEN, cur_init_ds_ep, cur_ds))

    name = 't_nB_LGD/test_1000_' + time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime(time.time()))
    np.savez(name, res_r, res_p, res_b, res_o, res_d)
