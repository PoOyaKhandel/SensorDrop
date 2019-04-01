import shelve
import os 
import numpy as np 

result_folder = "result"
path = result_folder + "/K="

def get_path(K):
    return path+str(K)+"/"

def gen_path(num):
    try:
        os.mkdir(result_folder)
    except Exception as e:
        print(e)
    try:
        os.chdir(result_folder)
    except Exception as e:
        print(e)
    k_l = np.round(np.linspace(0, 1, num), decimals=2)
    for k in k_l:
        try:
            os.mkdir("K="+str(k))
        except Exception as e:
            print(e)

## 

def save_action(K, data):
    d = shelve.open(path+str(K)+"/data K="+str(K))
    d["actions"] = data
    d.close()


def load_action(K):
    d = shelve.open(path+str(K)+"/data K="+str(K))
    b = d["actions"]
    d.close()
    return b

def save_acc(K, data):
    d = shelve.open(path+str(K)+"/data K="+str(K))
    d["acc"] = data
    d.close()


def load_acc(K):
    d = shelve.open(path+str(K)+"/data K="+str(K))
    b = d["acc"]
    d.close()
    return b

def save_reward(K, data):
    d = shelve.open(path+str(K)+"/data K="+str(K))
    d["reward"] = data
    d.close()


def load_reward(K):
    d = shelve.open(path+str(K)+"/data K="+str(K))
    b = d["reward"]
    d.close()
    return b

def load_data(Ks):
    datas = {}
    for K in Ks:    
        datas[str(K)]= {"rwd":load_reward(K), "acc":load_acc(K), "act":load_action(K)}
    return datas

if __name__ == "__main__":
    gen_path(11)
