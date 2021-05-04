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

def extract_(run_set,window):
    all_r = []
    all_smooth_r = []
    for run in run_set:
        r = api.run(run)
        data = r.history()
        rewards = data['Episode_Reward'].dropna()

        all_r.append(rewards)
        smooth_r = moving_average(rewards, window)
        all_smooth_r.append(smooth_r)


    return all_r, all_smooth_r

def data_viz(case_studies, labels):
    x = np.linspace(0, 1000000, len(rewards_mean))
    all_r_means = []
    all_err_value = []
    for case in case_studies:
        r_means = np.mean(case, axis=0)
        err_value = np.std(case, axis=0)
        all_r_means.append(r_means)
        all_err_value.append(err_value)
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

def compare_mnist(data):
    data_plot = []
    for digit in data:
        r_means = np.mean(digit, axis = 0)
        data_plot.append(r_means)

    labels = ['0','1','2','3','4','5','6','7','8','9']
    plt.xticks('Mnist Digits')
    plt.yticks('Average Episode Reward')

    plt.boxplot(data_plot,labels=labels)
    #plt.xticks('Mnist Digits')
    #plt.yticks('Average Episode Reward')
    plt.show()






if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = "6afc0a3c483f234dffd880c403a4aa3b488bc686"
    api = wandb.Api()

    #k = raw_data.keys().values
    #print(k)

    #print(len(rewards[0]))

    #rewards_mean = np.mean(rewards, axis=0)
    #err_value = np.std(rewards, axis=0)
    #print(len(rewards_mean))

    hierarchy_general = ["growspace/hierarchy/1em80av5","growspace/hierarchy/175xlpwl","growspace/hierarchy/sb1uoeaj"]
    hierarchy_easy = ["growspace/hierarchy/jueepcl0","growspace/hierarchy/3c8fbek9","growspace/hierarchy/15h341e6"]
    hierarchy_hard = ["growspace/hierarchy/1p9dxq57","growspace/hierarchy/1rcmk4bn","growspace/growspaceenv_baselines/3i9p841q"]

    control_general = ["growspace/control/2uw60tub","growspace/control/xufs0mnh","growspace/control/2k8ug0gs"]
    control_easy =["growspace/control/n2dzahl6","growspace/control/2i9wb6bu","growspace/growspaceenv_baselines/3n0dlkzl"]
    control_hard =["growspace/control/85prsz2s","growspace/control/27jlk0zv","growspace/control/1b3ym56i"]

    fairness_general =["","",""]
    fairness_easy =["growspace/fairness/nl2m3wyp","growspace/fairness/15e41fm9","growspace/fairness/18zho3qp"]
    fairness_middle =["growspace/fairness/2a0aa0hj","growspace/fairness/cugl7h51","growspace/fairness/1vkbn7na"]
    fairness_above =["growspace/fairness/g84ktb8c","growspace/fairness/1tgon55v","growspace/fairness/27yv69rl"]

    mnist0 =["growspace/mnistexperiments/e4q9dnu6","growspace/mnistexperiments/3vlmbhqm","growspace/mnistexperiments/1b6k8vfn"]
    mnist1 =["growspace/mnistexperiments/2kr2k6dm","growspace/mnistexperiments/6o6gvlc4","growspace/mnistexperiments/l18ag4ts"]
    mnist2 =["growspace/mnistexperiments/1zjuowgd","growspace/mnistexperiments/2utbbi0p","growspace/mnistexperiments/1gkup9v8"]
    mnist3 =["growspace/mnistexperiments/m27wfa1x","growspace/mnistexperiments/11o1sv4w","growspace/growspaceenv_baselines/2clvl10i"]
    mnist4 =["growspace/mnistexperiments/1qim4p5g","growspace/mnistexperiments/1ap6rfxi","growspace/mnistexperiments/2i6n0wua"]
    mnist5 =["growspace/mnistexperiments/5tkgqlz2","growspace/mnistexperiments/boeoqxyi","growspace/mnistexperiments/2hqwhjdi"]
    mnist6 =["growspace/mnistexperiments/2y01913k","growspace/mnistexperiments/2yl06la2","growspace/mnistexperiments/1e0tkqqe"]
    mnist7 =["growspace/mnistexperiments/20fondx6","growspace/mnistexperiments/actovp6d","growspace/mnistexperiments/1sty6m4b"]
    mnist8 =["growspace/mnistexperiments/1d348aky","growspace/mnistexperiments/2exzpp77","growspace/mnistexperiments/38guhriz"]
    mnist9 =["growspace/mnistexperiments/1aq1v5qv","growspace/mnistexperiments/2gshm6le","growspace/mnistexperiments/3rhg3jod"]

    r0, _ = extract_(mnist0,150)
    r1,_ = extract_(mnist1, 150)
    r2,_ = extract_(mnist2, 150)
    r3,_ = extract_(mnist3, 150)
    r4,_ = extract_(mnist4, 150)
    r5,_ = extract_(mnist5, 150)
    r6,_ = extract_(mnist6, 150)
    r7,_ = extract_(mnist7, 150)
    r8,_ = extract_(mnist0, 150)
    r9,_ = extract_(mnist0, 150)

    all_mnist_digits = [r0,r1,r2,r3,r4,r5,r6,r7,r8,r9]
    compare_mnist(all_mnist_digits)

    mnistmix =["growspace/mnistexperiments/1i3y6rha","",""]

