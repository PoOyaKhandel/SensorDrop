import save_result as sr
import numpy as np
import matplotlib.pyplot as plt

Ks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
    ax.plot(p[1], p[2], 'o-')
    ax.annotate(str(p[0]), (p[1], p[2]))

ax.grid()
ax.set_xlabel("accuracy")
ax.set_ylabel("overhead")
plt.show()