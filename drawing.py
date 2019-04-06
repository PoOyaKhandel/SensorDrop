import save_result as sr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

result_plot_font = 12

plot_font = 14
matplotlib.rcParams.update({'font.size': plot_font})

RKs = [0.5]
total_rw_ = [0.56]
total_acc_ = [0.98]

Ks = [0.1, 0.2, 0.4, 0.5,0.8, 0.9]
datas = sr.load_data(Ks)
print(datas.keys())
##Now we have all data :)
points = []

for k in Ks:
    k = str(k)
    a = np.argmax(datas[k]["rwd"][200:])
    acc = datas[k]["acc"][a]
    act = datas[k]["act"][a]
    rwd = datas[k]["rwd"][a]
    print("--"*50)
    print("k =",k, "acc =", acc, "act =", act, "rwd =", rwd)
    points.append((k, acc,act,rwd))    

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for p in points:
    ax.plot(p[1], p[2], 'p', markersize=plot_font+2)
    ax.annotate("K="+str(p[0]), (p[1]-0.002, p[2] + 0.01))

ax.grid()
ax.set_xlabel("accuracy", fontsize=plot_font)
ax.set_ylabel("overhead", fontsize=plot_font)
ax.tick_params(axis='both', which='major', labelsize=plot_font)
############drawing result

matplotlib.rcParams.update({'font.size': result_plot_font})

for k, total_acc, total_rew in zip(RKs, total_acc_, total_rw_):
    k = str(k)
    reward_history_mean = datas[k]["rwd"]
    accuracy_history_mean = datas[k]["acc"]
    action_history_mean = datas[k]["act"]
    ###################
    ave_fig = plt.figure()
    ax = ave_fig.add_subplot(2, 1, 1)
    # ax.set_title("a")
    ax.plot(reward_history_mean[200:], 'r-', label='reward')
    ax.legend()
    ax.plot(accuracy_history_mean[200:], 'k-', label='accuracy')
    ax.legend()
    ax.plot([total_acc for e in accuracy_history_mean[200:]], 'C5-.', label='without drop accuracy')
    ax.legend()
    ax.plot([total_rew for e in accuracy_history_mean[200:]], 'C9-.', label='without drop reward')
    ax.legend()
    #ax.set_ylim(0, 1.05)
    ax.set_xlabel('Iterations')
    ax.grid()

    # ave_fig2 = plt.figure()
    # ax2 = ave_fig2.add_subplot(1, 1, 1)
    ax = ave_fig.add_subplot(2, 1, 2)
    # ax.set_title("b")
    ax.plot(action_history_mean[200:], 'b-', label='communication overhead')
    ax.legend()
    ax.plot(accuracy_history_mean[200:], 'k-', label='accuracy')
    ax.legend()
    ax.plot([total_acc for e in accuracy_history_mean[200:]], 'C5-.', label='without drop accuracy')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Iterations')
    ave_fig.set_size_inches((11, 8.5), forward=False)

    #####draw overhead per iterations
    overhead_accuracy_fig = plt.figure()
    ax = overhead_accuracy_fig.add_subplot(1, 1, 1)
    colors = cm.rainbow(np.linspace(0, 1, len(accuracy_history_mean[200:])))
    i = 0
    for x, y, c in zip(accuracy_history_mean[200:], action_history_mean[200:], colors):
        if i%10 == 0 :
            ax.plot(x, y, '.',color=c)
        i = i + 1
    cmap = plt.get_cmap("rainbow")
    norm = plt.Normalize(0, len(colors))
    sm =  cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = overhead_accuracy_fig.colorbar(sm, ax=ax)
    cbar.ax.set_title("iterations")   
    ax.set_xlabel('accuracy')
    ax.set_ylabel('overhead')
    ax.grid()
    overhead_accuracy_fig.set_size_inches((11, 8.5), forward=False)

plt.show()
