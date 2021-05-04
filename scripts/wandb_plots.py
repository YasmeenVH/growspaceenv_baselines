import wandb
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import make_interp_spline, BSpline
#from scipy import interpolate.BSpline
#import seaborn as sns



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def extract_(run):
    return run



if __name__ == "__main__":
    #6afc0a3c483f234dffd880c403a4aa3b488bc686
    os.environ['WANDB_API_KEY'] = "6afc0a3c483f234dffd880c403a4aa3b488bc686"
    #wandb.init(settings=wandb.Settings(project='hierarchy', entity='growspace'))
    api = wandb.Api()
    run1_h_easy = api.run("growspace/hierarchy/jueepcl0")
    run2_h_easy = api.run("growspace/hierarchy/3cawbgcd")
    run3_h_easy = api.run("growspace/hierarchy/2y18ehdi")

    run1_h_general = api.run("growspace/hierarchy/1em80av5")
    run2_h_general = api.run("growspace/hierarchy/175xlpwl")
    run3_h_general = api.run("growspace/hierarchy/sb1uoeaj")

    raw_data = run1_h_general.history()
    raw_data2 = run2_h_general.history()
    raw_data3 = run3_h_general.history()
    k = raw_data.keys().values
    #print(k)
    rewards = []
    r_mean = raw_data['Episode_Reward'].dropna().values
    r_mean2 = raw_data2['Episode_Reward'].dropna().values
    r_mean3 = raw_data3['Episode_Reward'].dropna().values

    r1 = moving_average(r_mean,130)
    r2 = moving_average(r_mean2,130)
    r3 = moving_average(r_mean3,130)
    print('this is len of r_mean', len(r_mean))
    print('this is len of r_mean2', len(r_mean2))
    print('this is len of r_mean3', len(r_mean3))
    rewards.append(r1)
    rewards.append(r2)
    rewards.append(r3)

    #print(len(rewards[0]))

    rewards_mean = np.mean(rewards, axis=0)
    err_value = np.std(rewards, axis=0)
    #print(len(rewards_mean))
    x = np.linspace(0, 1000000, len(rewards_mean))
    #x_new = np.linspace(0,1000000, 900000)
    #a_BSpline = scipy.interpolate.make_interp_spline(x,rewards_mean)
    #y_new = a_BSpline(x_new)
    #plt.plot(x_new,y_new)
    #x2 = np.linspace(0, 1000000, len(r_mean2))
    #x3 = np.linspace(0, 1000000, len(r_mean3))
    #print('this is len of r_mean', len(r_mean))
    #print(r_mean[:10])
    fig, ax = plt.subplots(1)
    ax.plot(x, rewards_mean, color = 'green')
    #
    #ax.plot(x, av_random, color = 'magenta')
    ax.fill_between(x, rewards_mean+err_value, rewards_mean-err_value, alpha=0.5, color='green')
    # ax.fill_between(x, av_random + err_random, av_random - err_random, alpha=0.5, color='magenta')
    ax.set_title(r'Average Episode Eewards')
    ax.set_ylabel("rewards")
    ax.set_xlabel("steps")
    #plt.plot(x, rewards_mean)
    plt.show()
    hierarchy_general = ["growspace/hierarchy/1em80av5","growspace/hierarchy/175xlpwl","growspace/hierarchy/sb1uoeaj"]
    hierarchy_easy = ["","",""]
    hierarchy_hard = ["","",""]

    control_general = ["growspace/control/2uw60tub","growspace/control/xufs0mnh","growspace/control/2k8ug0gs"]
    control_easy =["growspace/control/n2dzahl6","growspace/control/2i9wb6bu",""]
    control_hard =["growspace/control/85prsz2s","growspace/control/27jlk0zv","growspace/control/1b3ym56i"]

    fairness_general =["","",""]
    fairness_easy =["","",""]
    fairness_middle =["","",""]
    fairness_hard =["","",""]

    mnist0 =["growspace/mnistexperiments/e4q9dnu6","growspace/mnistexperiments/3vlmbhqm","growspace/mnistexperiments/1b6k8vfn"]
    mnist1 =["growspace/mnistexperiments/2kr2k6dm","growspace/mnistexperiments/6o6gvlc4","growspace/mnistexperiments/l18ag4ts"]
    mnist2 =["growspace/mnistexperiments/1zjuowgd","growspace/mnistexperiments/2utbbi0p","growspace/mnistexperiments/1gkup9v8"]
    mnist3 =["growspace/mnistexperiments/1udhtt4s","",""]
    mnist4 =["growspace/mnistexperiments/1qim4p5g","growspace/mnistexperiments/1ap6rfxi","growspace/mnistexperiments/2i6n0wua"]
    mnist5 =["growspace/mnistexperiments/5tkgqlz2","growspace/mnistexperiments/boeoqxyi","growspace/mnistexperiments/2hqwhjdi"]
    mnist6 =["growspace/mnistexperiments/2y01913k","growspace/mnistexperiments/2yl06la2","growspace/mnistexperiments/1e0tkqqe"]
    mnist7 =["growspace/mnistexperiments/20fondx6","growspace/mnistexperiments/actovp6d","growspace/mnistexperiments/1sty6m4b"]
    mnist8 =["growspace/mnistexperiments/1d348aky","growspace/mnistexperiments/2exzpp77","growspace/mnistexperiments/38guhriz"]
    mnist9 =["growspace/mnistexperiments/1aq1v5qv","growspace/mnistexperiments/2gshm6le","growspace/mnistexperiments/3rhg3jod"]
    mnistmix =["growspace/mnistexperiments/1i3y6rha","",""]

