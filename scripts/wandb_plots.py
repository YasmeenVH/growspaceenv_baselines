import wandb
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
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
        rewards = data['Reward Mean'].dropna()

        all_r.append(rewards)
        smooth_r = moving_average(rewards, window)
        all_smooth_r.append(smooth_r)


    return all_r, all_smooth_r

def other_data(run_set):
    all_actions = []
    all_branches = []
    for run in run_set:
        r = api.run(run)
        data = r.history()
        actions = data['Discrete Actions'].dropna()
        branches = data['Number of Mean New Branches'].dropna()
        all_branches.append(branches)
        all_actions.append(actions)
        #smooth_r = moving_average(rewards, window)
        #all_smooth_r.append(smooth_r)

    return all_actions, all_branches

def other_branches(run_set):
    all_branches2 = []
    all_branches1 = []
    for run in run_set:
        r = api.run(run)
        data = r.history()
        branches1 = data['Number of Mean New Branches'].dropna()
        #branches2 = data['Number of Mean New Branches of Plant 2'].dropna()
        all_branches1.append(branches1)
        #all_branches2.append(branches2)

    val = np.linspace(0, 230, 24)
    #print(all_branches1[0],'len of b1')
    all_b1 = []
    for case in all_branches1:
        bs = []
        for idx in val:
            bs.append(case[int(idx)])
            all_b1.append(bs)
    all_b2 = []
    # for case in all_branches2:
    #     bs = []
    #     for idx in val:
    #         bs.append(case[int(idx)])
    #         all_b2.append(bs)



    #b1 = np.mean(all_b1, axis=0)
    #b2 = np.mean(all_b2, axis=0)
    # data_plot = [b1,b2]
    # fig, ax = plt.subplots(1)
    # # .xticks('Mnist Digits')Episode Reward
    # ax.set_ylabel('Average Number of Branches')
    # ax.set_xlabel('Plants')
    # ax.set_title('Branching of Plants')
    # labels = ['Plant 1', 'Plant 2']
    # #data_plot = [left, right, increase, decrease, pass_]
    # ax.boxplot(data_plot, labels=labels, patch_artist=True)
    # # plt.xticks('Mnist Digits')
    # # plt.yticks('Average Episode Reward')
    # plt.show()

    #err = np.std(1, axis=0)
        #
    hb1 = np.mean(all_b1, axis=0)
    err1 = np.std(all_b1, axis=0)
        # #h, err = data_viz(all_b)
        #smooth_r = moving_average(rewards, window)
        #all_smooth_r.append(smooth_r

    return hb1, err1

def data_viz(case_studies):
    all_r_means = []
    all_err_value = []
    for case in case_studies:
        print('len of case',len(case))
        r_means = np.mean(case, axis=0)
        err_value = np.std(case, axis=0)
        all_r_means.append(r_means)
        all_err_value.append(err_value)

    return all_r_means, all_err_value

def compare_mnist(data):
    data_plot = []
    for digit in data:
        r_means = np.mean(digit, axis = 0)
        data_plot.append(r_means)

    labels = ['0','1','2','3','4','5','6','7','8','9']
    fig, ax = plt.subplots(1)
    #.xticks('Mnist Digits')
    ax.set_ylabel('Average Episode Reward')
    ax.set_xlabel('Mnist digits')
    ax.boxplot(data_plot,labels=labels, patch_artist=True)
    #plt.xticks('Mnist Digits')
    #plt.yticks('Average Episode Reward')
    plt.show()






if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = "6afc0a3c483f234dffd880c403a4aa3b488bc686"
    api = wandb.Api()
    analysis = 'ch_branch'
    #k = raw_data.keys().values
    #print(k)
    #hierarchy_general = ["growspace/hierarchy/1em80av5", "growspace/hierarchy/175xlpwl", "growspace/hierarchy/sb1uoeaj"]
    #actions, branches = other_data(hierarchy_general)
    if analysis == 'fairbranch':
        fairness_easy = ["growspace/fairness/nl2m3wyp", "growspace/fairness/15e41fm9", "growspace/fairness/18zho3qp"]
        fairness_middle =["growspace/fairness/2a0aa0hj","growspace/fairness/cugl7h51","growspace/fairness/1vkbn7na"]
        fairness_above =["growspace/fairness/g84ktb8c","growspace/fairness/1tgon55v","growspace/fairness/27yv69rl"]
        other_branches(fairness_above)
    if analysis == 'actions':
        hierarchy_easy = ["growspace/hierarchy/jueepcl0", "growspace/hierarchy/3c8fbek9", "growspace/hierarchy/15h341e6"]

        hierarchy_hard = ["growspace/hierarchy/1p9dxq57","growspace/hierarchy/1rcmk4bn","growspace/growspaceenv_baselines/3i9p841q"]
        control_easy =["growspace/control/n2dzahl6","growspace/control/2i9wb6bu"]#,"growspace/growspaceenv_baselines/3n0dlkzl"]
        control_hard =["growspace/control/85prsz2s","growspace/control/27jlk0zv","growspace/control/1b3ym56i"]
        actions, branches = other_data(hierarchy_easy)
        actions1, branches1 = other_data(hierarchy_hard)
        print("what is actions", actions[1][0]['values'])
        print(actions1)
        val = np.linspace(0, 230, 24)
        all_act = []
        for case in actions:
            act = []
            for idx in val:
                ma_data = case[idx]['values']
                act.append(ma_data)
            all_act.append(act)
        #print(len(branches1[0]), 'what is length of branhces')
        mean_act = np.mean(all_act, axis=0)
        #print(mean_act, "mean act")
        good_idx = [0,2,5,7,9]
        action_all5 = []
        for x in mean_act:
            goodacts = []
            for id in good_idx:
                goodacts.append(x[id])
            action_all5.append(goodacts)
        print(action_all5, 'actionall5')
        left = []
        right = []
        increase = []
        decrease = []
        pass_ = []
        for x in action_all5:
            left.append(x[0])
            right.append(x[1])
            increase.append(x[2])
            decrease.append(x[3])
            pass_.append(x[4])

        fig, ax = plt.subplots(1)
        # .xticks('Mnist Digits')Episode Reward
        ax.set_ylabel('Average steps per action')
        ax.set_xlabel('Actions')
        ax.set_title('Action Selection')
        labels = ['Left','Right','Increase','Decrease','Pass']
        data_plot = [left,right,increase,decrease,pass_]
        ax.boxplot(data_plot, labels=labels, patch_artist=True)
        # plt.xticks('Mnist Digits')
        # plt.yticks('Average Episode Reward')
        plt.show()
        # create plot
        #fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
        #fig, ax = plt.subplot()
        #fig.suptitle('Action Selection')
        #index = np.arange(n_groups)
        val333 = list(range(1,41))
        val333 = [float(i) for i in val333]
        print(val333, "what is val333")
        #val_str =
        bar_width = 0.2
        opacity = 0.8
        index = np.arange(40).astype(int)
        # plt.hist([left, right, increase, decrease, pass_], stacked=True,
        #            label=['Left','Right','Increase','Decrease','Pass'])
        # d_left = gaussian_kde(left)
        # d_right = gaussian_kde(right)
        # d_increase = gaussian_kde(increase)
        # d_decrease =gaussian_kde(decrease)
        # d_pass = gaussian_kde(pass_)
        # #
        # ax1.bar(index, left, bar_width,
        #                  alpha=opacity,
        #                  color='b',
        #                  label='Left')
        #
        # ax2.bar(index+ bar_width, right, bar_width,
        #                  alpha=opacity,
        #                  color='g',
        #                  label='Right')
        # ax3.bar(index + bar_width*2, increase, bar_width,
        #                  alpha=opacity,
        #                  color='r',
        #                  label='Increase')
        # ax4.bar(index + bar_width*3, decrease, bar_width,
        #                  alpha=opacity,
        #                  color='y',
        #                  label='Decrease')
        # ax5.bar(index + bar_width*4, pass_, bar_width,
        #                  alpha=opacity,
        #                  color='m',
        #                  label='Stay')

        # left = moving_average(left,10)
        # right = moving_average(right,10)
        # increase =moving_average(increase,10)
        # decrease = moving_average(decrease, 10)
        # pass_ = moving_average(pass_, 10)
        # index = np.linspace(1,40,len(pass_))
        # plt.figure(figsize=(17,4))
        # #fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
        #print(len(leftt),'len of leftt')
        # plt.bar(index,left,bar_width,color = 'b',label='Left')
        # plt.bar(index+bar_width, right,bar_width, color='g',label='Right')
        # plt.bar(index+bar_width*2, increase,bar_width, color='c', label='Increase')
        # plt.bar(index+bar_width*3, decrease,bar_width, color='r',label='Decrease')
        # plt.bar(index+bar_width*4, pass_,bar_width, color='y',label='Pass')
        #
        # plt.xlim([0, 42])
        # plt.ylim([0,620])
        # plt.xlabel('PPO updates')
        # plt.ylabel('Number of Steps per update')
        # plt.title('Action Selection')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        # # plt.tight_layout()
        # # #plt.xticks(index + bar_width, val333)
        # #plt.legend()
        # #plt.show()
        # # for ax in fig.get_axes():
        # #     ax.label_outer()
        # plt.show()
        # val = np.linspace(0,390,40)
        # all_b = []
        # for case in branches:
        #     bs = []
        #     for idx in val:
        #         bs.append(case[int(idx)])
        #     all_b.append(bs)
        # all_b1 = []
        # for case in branches1:
        #     bs = []
        #     for idx in val:
        #         bs.append(case[int(idx)])
        #     all_b1.append(bs)


        # hb = np.mean(all_b, axis=0)
        # err = np.std(all_b, axis=0)
        #
        # hb1 = np.mean(all_b1, axis=0)
        # err1 = np.std(all_b1, axis=0)
        # #h, err = data_viz(all_b)
        # x = np.linspace(0,1000000,40)
        # fig, ax = plt.subplots(1)
        #
        #
        # labels = ['New Branches 1','New Branches 2']
        # fig, ax = plt.subplots(1)
        # #.xticks('Mnist Digits')
        # ax.set_ylabel('Average New Branches')
        # ax.set_xlabel('Plants')
        # data_plot = [hb, hb1]
        # ax.boxplot(data_plot,labels=labels, patch_artist=True)
        # #ax.plot(x, hb, color='green')
        # #ax.fill_between(x, hb + err, hb - err, alpha=0.5, color='green', label='Branches Easy')
        #
        # ax.plot(x, hb1, color='blue')
        # ax.fill_between(x, hb1 + err1, hb1 - err1, alpha=0.5, color='blue', label='Branches Hard')
        # ax.set_ylabel("# Branches")
        # ax.set_xlabel("Steps")
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        #
        # plt.show()
        #


        #print(actions[0][20])#[1])
        #print(branches[0])

    if analysis=='hierarchy':
        hierarchy_general = ["growspace/hierarchy/1em80av5","growspace/hierarchy/175xlpwl","growspace/hierarchy/sb1uoeaj"]
        hierarchy_easy = ["growspace/hierarchy/jueepcl0","growspace/hierarchy/3c8fbek9","growspace/hierarchy/15h341e6"]
        hierarchy_hard = ["growspace/hierarchy/1p9dxq57","growspace/hierarchy/1rcmk4bn","growspace/growspaceenv_baselines/3i9p841q"]
        hard_oracle = [0.29,0.29,0.29]
        hard_random = [0.074,0.074,0.074]
        easy_oracle = [1.07,1.07,1.07]
        easy_random = [0.25,0.25,0.25]

        r_easy, r_easy_smooth = extract_(hierarchy_easy, 150)
        r_hard, r_hard_smooth = extract_(hierarchy_hard, 150)
        # for i in range(len(r_easy_smooth)):
        #     print(len(r_easy_smooth[i]))
        #     print(len(r_hard_smooth[i]))
        all_hierarchy = [r_easy_smooth, r_hard_smooth]
        h, err = data_viz(all_hierarchy)
        easy, err_e = h[0], err[0]
        hard, err_h = h[1], err[1]

        x = np.linspace(0, 1000000, len(easy))
        fig, ax = plt.subplots(1)
        ax.plot(x, easy, color='green')
        ax.fill_between(x, easy + err_e, easy - err_e, alpha=0.5, color='green', label='Easy')
        ax.plot(x, hard, color='blue')
        ax.fill_between(x, hard + err_h, hard - err_h, alpha=0.5, color='blue', label='Hard')

        x_oracle = np.linspace(0, 1000000, 3)
        ax.plot(x_oracle, easy_oracle, color='green', linestyle='dotted', label='Easy Oracle')
        ax.plot(x_oracle, easy_random, color='green', linestyle='dashed', label='Easy Random')

        ax.plot(x_oracle, hard_oracle, color='blue', linestyle='dotted', label='Hard Oracle')
        ax.plot(x_oracle, hard_random, color='blue', linestyle='dashed', label='Hard Random')

        # ax.plot()
        # ax.fill_between(x, av_random + err_random, av_random - err_random, alpha=0.5, color='magenta')
        # ax.set_title(r'Average Episode Rewards')
        ax.set_ylabel("Rewards")
        ax.set_xlabel("Steps")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        # plt.plot(x, rewards_mean)
        plt.show()

    if analysis=='control':
        control_general = ["growspace/control/2uw60tub","growspace/control/xufs0mnh","growspace/control/2k8ug0gs"]
        control_easy =["growspace/control/n2dzahl6","growspace/control/2i9wb6bu"]#,"growspace/growspaceenv_baselines/3n0dlkzl"]
        control_hard =["growspace/control/85prsz2s","growspace/control/27jlk0zv","growspace/control/1b3ym56i"]
        hard_oracle = [0.36,0.36,0.36]
        hard_random = [0.11,0.11,0.11]
        easy_oracle = [1.2,1.2,1.2]
        easy_random = [0.42,0.42,0.42]


        r_easy,r_easy_smooth = extract_(control_easy, 50)
        r_hard, r_hard_smooth = extract_(control_hard, 50)
        # for i in range(len(r_easy_smooth)):
        #     print(len(r_easy_smooth[i]))
        #     print(len(r_hard_smooth[i]))
        all_control = [r_easy_smooth,r_hard_smooth]
        control, err = data_viz(all_control)
        easy, err_e = control[0],err[0]
        hard, err_h = control[1],err[1]

        x = np.linspace(0, 1000000, len(easy))
        fig, ax = plt.subplots(1)
        ax.plot(x, easy, color='green')
        ax.fill_between(x, easy + err_e, easy - err_e, alpha=0.5, color='green', label='Easy')
        ax.plot(x,hard,color='blue')
        ax.fill_between(x, hard+ err_h, hard - err_h,alpha=0.5, color='blue',label='Hard')

        x_oracle = np.linspace(0,1000000, 3)
        ax.plot(x_oracle,easy_oracle,color='green', linestyle='dotted',label='Easy Oracle')
        ax.plot(x_oracle, easy_random, color='green', linestyle='dashed', label='Easy Random')

        ax.plot(x_oracle,hard_oracle,color='blue', linestyle='dotted',label='Hard Oracle')
        ax.plot(x_oracle, hard_random, color='blue', linestyle='dashed',label='Hard Random')

        #ax.plot()
        # ax.fill_between(x, av_random + err_random, av_random - err_random, alpha=0.5, color='magenta')
        # ax.set_title(r'Average Episode Rewards')
        ax.set_ylabel("Rewards")
        ax.set_xlabel("Steps")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        # plt.plot(x, rewards_mean)
        plt.show()

    if analysis=='fairness':
        fairness_general =["","",""]
        fairness_easy =["growspace/fairness/nl2m3wyp","growspace/fairness/15e41fm9","growspace/fairness/18zho3qp"]
        fairness_middle =["growspace/fairness/2a0aa0hj","growspace/fairness/cugl7h51","growspace/fairness/1vkbn7na"]
        fairness_above =["growspace/fairness/g84ktb8c","growspace/fairness/1tgon55v","growspace/fairness/27yv69rl"]
        above_oracle =[0.09,0.09,0.09]
        above_random =[0.04,0.04,0.04]
        middle_oracle =[0.12,0.12,0.12]
        middle_random =[0.055,0.055,0.055]
        easy_oracle =[0.2,0.2,0.2]
        easy_random =[0.1,0.1,0.1]

        r_easy,r_easy_smooth = extract_(fairness_easy, 200)
        r_middle, r_middle_smooth = extract_(fairness_middle, 200)
        r_above, r_above_smooth = extract_(fairness_above, 200)

        smooth_r = moving_average(rewards, window)
        all_smooth_r.append(smooth_r)

        all_fair = [r_easy_smooth,r_middle_smooth,r_above_smooth]
        fair, err = data_viz(all_fair)
        easy, err_e = fair[0],err[0] #data_viz(r_easy_smooth)
        middle, err_m = fair[1],err[1] #data_viz(r_middle_smooth)
        above, err_a = fair[2],err[2] #data_viz(r_above_smooth)

        x = np.linspace(0, 1000000, len(easy))
        fig, ax = plt.subplots(1)
        ax.plot(x, easy, color='green')
        ax.fill_between(x, easy + err_e, easy - err_e, alpha=0.5, color='green', label='Easy')
        ax.plot(x,middle,color='blue')
        ax.fill_between(x, middle + err_m, middle - err_m,alpha=0.5, color='blue',label='Middle')
        ax.plot(x,above,color='purple')
        ax.fill_between(x, above + err_a, above - err_a, alpha=0.5, color='purple', label='Above')
        x_oracle = np.linspace(0,1000000, 3)
        ax.plot(x_oracle,easy_oracle,color='green', linestyle='dotted',label='Easy Oracle')
        ax.plot(x_oracle, easy_random, color='green', linestyle='dashed', label='Easy Random')

        ax.plot(x_oracle,middle_oracle,color='blue', linestyle='dotted',label='Middle Oracle')
        ax.plot(x_oracle, middle_random, color='blue', linestyle='dashed',label='Middle Random')

        ax.plot(x_oracle,above_oracle,color='purple', linestyle='dotted',label='Above Oracle')
        ax.plot(x_oracle, above_random, color='purple', linestyle='dashed',label='Above Random')

        #ax.plot()
        # ax.fill_between(x, av_random + err_random, av_random - err_random, alpha=0.5, color='magenta')
        # ax.set_title(r'Average Episode Rewards')
        ax.set_ylabel("Rewards")
        ax.set_xlabel("Steps")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        # plt.plot(x, rewards_mean)
        plt.show()

    if analysis =='mnist':
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

    if analysis=='mnistmix':
        mnistmix =["growspace/mnistexperiments/1i3y6rha","growspace/mnistexperiments/3onp6261"]
        mnist0 =["growspace/mnistexperiments/e4q9dnu6","growspace/mnistexperiments/3vlmbhqm","growspace/mnistexperiments/1b6k8vfn"]
        mnist3 =["growspace/mnistexperiments/m27wfa1x","growspace/mnistexperiments/11o1sv4w","growspace/growspaceenv_baselines/2clvl10i"]
        #r0, mnist00 = extract_(mnist0, 150)
        #r3, mnist33 = extract_(mnist3, 150)
        mnist_curriculum = ["growspace/mnistexperiments/t1f7ovsx","growspace/mnistexperiments/4o5ulndv"]

        r_easy,r_easy_smooth = extract_(mnistmix, 15)
        r_hard, r_hard_smooth = extract_(mnist_curriculum, 15)
        # for i in range(len(r_easy_smooth)):
        #     print(len(r_easy_smooth[i]))
        #     print(len(r_hard_smooth[i]))
        all_control = [r_easy_smooth,r_hard_smooth]
        control, err = data_viz(all_control)
        easy, err_e = control[0],err[0]
        hard, err_h = control[1],err[1]

        hard_oracle = [9.5, 9.5, 9.5]
        hard_random = [4.2, 4.2, 4.2]
        easy_oracle = [10.2, 10.2, 10.2]
        easy_random = [3.36, 3.36, 3.36]
        print(len(hard), 'what is length')
        x = np.linspace(0, 1000000, len(easy))
        x2 = np.linspace(0,1000000,len(hard))
        fig, ax = plt.subplots(1)
        ax.plot(x, easy, color='green')
        ax.fill_between(x, easy + err_e, easy - err_e, alpha=0.5, color='green', label='Mix')
        ax.plot(x2,hard,color='blue')
        ax.fill_between(x2, hard+ err_h, hard - err_h,alpha=0.5, color='blue',label='Curriculum')

        #x_oracle = np.linspace(0,1000000, 3)
        #ax.plot(x_oracle,easy_oracle,color='green', linestyle='dotted',label='Mnist 3 Oracle')
        #ax.plot(x_oracle, easy_random, color='green', linestyle='dashed', label='Mnist 3 Random')

        #ax.plot(x_oracle,hard_oracle,color='blue', linestyle='dotted',label='Mnist 0 Oracle')
        #ax.plot(x_oracle, hard_random, color='blue', linestyle='dashed',label='Mnist 0 Random')

        #ax.plot()
        # ax.fill_between(x, av_random + err_random, av_random - err_random, alpha=0.5, color='magenta')
        # ax.set_title(r'Average Episode Rewards')
        ax.set_ylabel("Rewards")
        ax.set_xlabel("Steps")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        # plt.plot(x, rewards_mean)
        plt.show()

    if analysis == 'ch_branch':
        hierarchy_easy = ["growspace/hierarchy/jueepcl0", "growspace/hierarchy/3c8fbek9", "growspace/hierarchy/15h341e6"]

        hierarchy_hard = ["growspace/hierarchy/1p9dxq57","growspace/hierarchy/1rcmk4bn","growspace/growspaceenv_baselines/3i9p841q"]
        control_easy =["growspace/control/n2dzahl6","growspace/control/2i9wb6bu"]#,"growspace/growspaceenv_baselines/3n0dlkzl"]
        control_hard =["growspace/control/85prsz2s","growspace/control/27jlk0zv","growspace/control/1b3ym56i"]

        c_e_b, c_e_e = other_branches(control_easy)
        c_h_b, c_h_e = other_branches(control_hard)

        x = np.linspace(0, 1000000, len(c_e_b))
        x2 = np.linspace(0,1000000,len(c_h_b))
        fig, ax = plt.subplots(1)
        ax.plot(x, c_e_b, color='green')
        ax.fill_between(x, c_e_b + c_e_e, c_e_b - c_e_e, alpha=0.5, color='green', label='Mix')
        ax.plot(x2,c_h_b,color='blue')
        ax.fill_between(x2, c_h_b + c_h_e, c_h_b - c_h_e,alpha=0.5, color='blue',label='Curriculum')
        plt.show()