import termcolor
import numpy as np
import os
import h5py
import math
import os.path
import csv
import time
import illustris_python as il
import matplotlib.pyplot as plt

def log10(x):
    if x > 0:
        return math.log10(x)
    else:
        return np.nan

def box_smooth(data_array):
    N = len(data_array)

    data_smooth = []

    for i in range(0, N):
        data_i = data_array[int(np.maximum(i - 1, 0)):int(np.minimum(i + 2, N))]
        # print(np.nanmean(data_i))

        data_smooth.append(np.nanmean(data_i))

    data_smooth = np.array(data_smooth).ravel()
    data_smooth[0] = np.nanmedian(data_array[:1])
    return data_smooth

def bootstrap_scatter_err(samples):
    mask_finite = np.isfinite(samples)
    samples = samples[mask_finite]
    index_all = range(len(samples))
    err_all = []
    N=100
    for i in range(0,N):
        index_choose = np.random.randint(0,len(samples)-1,len(samples))
        # k_i = np.nanstd(samples[index_choose])
        k_i = np.percentile(samples[index_choose],84)-np.percentile(samples[index_choose],16)
        k_i = k_i/2
        err_all.append(k_i)
    err_all = np.array(err_all)
    if len(samples)<0:
        err_all = np.nan

    return err_all


def exp(x):
    try:
        return math.exp(x)
    except:
        return np.inf

def Mpeak_log_to_Vpeak_log(Mpeak_log):
    return 0.3349*Mpeak_log - 1.672

G = 4.301 * 10 ** (-9)
cons = (4 * G * np.pi / (3 * (1 / 24 / (1.5 * 10 ** (11))) ** (1 / 3))) ** 0.5


def calculate_v_dispersion(Mh):
    return Mh ** (1 / 3) * cons


exp = np.vectorize(exp)
log10 = np.vectorize(log10)



plot_path = "/Users/caojunzhi/Downloads/upload_201907_Jeremy/"


if os.path.isdir("/Volumes/SSHD_2TB") == True:
    print("The code is on Spear of Adun")

    ## Move to Data_10TB
    data_path = "/Volumes/Data_10TB/"

elif os.path.isdir("/mount/sirocco1/jc6933/test") == True:
    data_path = "/mount/sirocco2/jc6933/Data_sirocco/"
    print("The code is on Sirocco")

# Kratos
elif os.path.isdir("/home/jc6933/test_kratos") == True:
    data_path = "/mount/kratos/jc6933/Data/"
    print("The code is on Kratos")

# Void Seeker
elif os.path.isdir("/home/jc6933/test_Void_Seeker") == True:
    data_path = "/mount/Void_Seeker/Data_remote/"
    print("The code is on Void Seeker")

### PRINCE:
elif os.path.isdir("/home/jc6933/test_folder_prince") == True:
    data_path = "/scratch/jc6933/"
    print("The code is on PRINCE")

else:
    print("The code is on local")
    data_path = "/Volumes/Extreme_SSD/Data/"

print("data_path %s" % data_path)

n_group = 99

######## trace from merger tree files:


Mh_target_log = np.linspace(10, 13, 18)

##!! extend the Mh_target a little bit but with bigger bin size for high Mh:
Mh_target_log_high = np.linspace(13, 14.5, 7)

Mh_target_log = np.append(Mh_target_log, Mh_target_log_high)

binsize = 0.05

## define a function which returns scatter at input redshift:

n_group = 99



N_snap_shot = []
N_redshift = []

with open(data_path+"TNG100/TNG100-1/url/" + "TNG100_information_csv.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for i, row in enumerate(csv_reader):
        if i > 2:
            N_snap_shot.append(row[1])
            N_redshift.append(row[2])

N_snap_shot = np.array(N_snap_shot, dtype=int)
N_redshift = np.array(N_redshift, dtype=float)

########### merger tree:
### read the group catalog:
# fields = ['GroupBHMass', "Group_M_Crit200", "GroupFirstSub"]

basePath = data_path + 'TNG100/TNG300-1/output'


fields = ['SubhaloMass', 'SubfindID', 'SnapNum', 'SubhaloStellarPhotometricsMassInRad', 'SubhaloVmax',"SubhaloBHMass"]
# N_tot = 4371210


N_spike_array = []

Mh_log_all = []
Vmax_log_all = []
BH_mass_all = []

speed = 0
time_previous = 0

# use linear interpolation to interpolate them onto the same N_snap
N_tot=20000


N_tot=1000


for i in range(0,N_tot):

    if i % 1000 == 0:
        speed = (100 / (time.time() - time_previous))

        print("sample per second=%.2f" % speed)

        print("Doing %d of %d, time left=%.2f seconds" % (i, N_tot, (N_tot - i) / speed))
        time_previous = time.time()

    try:
        tree = il.sublink.loadTree(basePath, n_group, i, fields=fields, onlyMPB=True)
        x = tree['SnapNum']
        # Ms_log = log10(tree['SubhaloStellarPhotometricsMassInRad'] * 1e10 / 0.704)
        Mh_log = log10(tree['SubhaloMass'] * 1e10 / 0.704)
        Mh_log = np.concatenate((Mh_log, np.nanmin(Mh_log) * np.ones(100 - len(Mh_log))))
        Vmax_log = log10(tree['SubhaloVmax'])
        Vmax_log = np.concatenate((Vmax_log, np.nanmin(Vmax_log) * np.ones(100 - len(Vmax_log))))
        BH_mass_log = log10(tree["SubhaloBHMass"] * 1e10 / 0.704)
        BH_mass_log = np.concatenate((BH_mass_log, np.nanmin(BH_mass_log) * np.ones(100 - len(BH_mass_log))))

        mask_Mh = np.isfinite(Mh_log)
        # drop samples with less than 80 finite numbers:
        if len(Mh_log[mask_Mh])<80:
            pass
        else:
            Mh_log_all.append(Mh_log)
            Vmax_log_all.append(Vmax_log)
            BH_mass_all.append(BH_mass_log)
    except:
        # no data available set N_spike=-1
        print("this one fails %d"%i)

Mh_log_all = np.array(Mh_log_all)
Vmax_log_all = np.array(Vmax_log_all)
BH_mass_all = np.array(BH_mass_all)

################
## build LSTM neural network with keras:
from pandas import read_csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# normalize
def minmax_normalize(X):
    # X N_smaple *N_time_step
    X = X.T
    X = (X - np.nanmin(X, axis=0)) / (np.nanmax(X, axis=0) - np.nanmin(X, axis=0))
    return X.T

Mh_log_all_scaled = minmax_normalize(Mh_log_all)
Vmax_log_all_scaled = minmax_normalize(Vmax_log_all)
BH_mass_all_scaled = minmax_normalize(BH_mass_all)

Mh_log_all_T = Mh_log_all.T
Mh_min_array = np.nanmin(Mh_log_all_T, axis=0)
Mh_max_array = np.nanmax(Mh_log_all_T, axis=0)

## remember the nanmin for Mh_log to scale it back:


## fusion:
# here we use time_lag= 1 snap_num. May change in future.
fusion = np.dstack((Mh_log_all_scaled[:,:-1],Vmax_log_all_scaled[:,:-1],BH_mass_all_scaled[:,:-1],Mh_log_all_scaled[:,1:]))
fusion.shape

# time array

time_array = list(range(99))

# split train test:
index = np.where(np.isnan(fusion))
fusion[index] = 0


# save in pickle
import pickle
pickle.dump(fusion,open(data_path+"LSTM_merger/fusion.pkl","wb"))


### can use 80% of the data to predict the redshift for next epoch

#n_train_hours = int(0.8*len(time_array))
n_train=int(0.8*fusion.shape[0])
# (35039, 1, 8) (35039,)
X_train = fusion[:n_train,:,:-1]
X_train = X_train.reshape([-1,1,3])

y_train = fusion[:n_train,:,-1]
y_train = y_train.ravel()


# test:
X_test = fusion[n_train:,:,:-1]
X_test = X_test.reshape([-1,1,3])

y_test = fusion[n_train:,:,-1]
y_test = y_test.ravel()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

## build LSTM:
import tensorflow as tf
# design network
model = tf.keras.Sequential([
  tf.keras.layers.LSTM(50,input_shape=(X_train.shape[1],X_train.shape[2])),  # must declare input shape
  tf.keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'acc'])
model.summary()

# train:
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=False)
plt.plot(history.history["acc"],"k")




# predict the Mh value for next time step:
y_predict = model.predict(X_test)
y_predict = y_predict.reshape([-1,99])

y_true = y_test.reshape([-1,99])

### de-normalize:
y_predict = y_predict.T
y_predict = y_predict*(Mh_max_array[n_train:]-Mh_min_array[n_train:])+Mh_min_array[n_train:]
y_predict = y_predict.T

y_true = y_true.T
y_true = y_true*(Mh_max_array[n_train:]-Mh_min_array[n_train:])+Mh_min_array[n_train:]
y_true = y_true.T


import sklearn
import matplotlib.pyplot as plt
from matplotlib.pylab import rc
import matplotlib

plot_path = "/Users/caojunzhi/Downloads/upload_201907_Jeremy/"


# plot

### plot scatter:
font = {'family': 'normal',
        'weight': 'bold',
        'size': 25}

matplotlib.rc('font', **font)

fig, axes = plt.subplots(1, 1, squeeze=False)

rc('axes', linewidth=3)

ax = plt.subplot(1,1,1)


color_array = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


plt.plot(y_true.ravel(),y_predict.ravel(),"k.",label="R^2=%.2f"%sklearn.metrics.r2_score(y_true.ravel(),y_predict.ravel()))

plt.legend()


plt.xlabel(r"$\log[M_h]$ True", fontweight='bold', fontsize=25)
plt.ylabel(r"$\log[M_h]$ Predicted", fontweight='bold', fontsize=25)

#axes[0, 0] = plt.gca()
#axes[0, 0].set_ylim([0, 0.7])

plt.tick_params(which='both', width=2, direction="in")
plt.tick_params(which='major', length=7, direction="in")
plt.tick_params(which='minor', length=4, color='k', direction="in")

plt.minorticks_on()
plt.title(r"log[Mh] predicted from LSTM")

plt.legend(prop={'size': 18})



fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(12,12)

save_path = plot_path + "Mh_log_LSTM_TNG100" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()
print("Done")




### histogram of difference:

### plot scatter:
font = {'family': 'normal',
        'weight': 'bold',
        'size': 25}

matplotlib.rc('font', **font)

fig, axes = plt.subplots(1, 1, squeeze=False)

rc('axes', linewidth=3)

ax = plt.subplot(1,1,1)


color_array = ['b', 'g', 'r', 'c', 'm', 'y', 'k']



plt.hist(y_true.ravel()-y_predict.ravel(),bins=np.linspace(-0.25,0.25,12),label="N_tot=20000")
#plt.xlabel("Difference in dex")



plt.xlabel("Difference in dex", fontweight='bold', fontsize=25)
#plt.ylabel(r"$\log[M_h]$ Predicted", fontweight='bold', fontsize=25)

#axes[0, 0] = plt.gca()
#axes[0, 0].set_ylim([0, 0.7])

plt.tick_params(which='both', width=2, direction="in")
plt.tick_params(which='major', length=7, direction="in")
plt.tick_params(which='minor', length=4, color='k', direction="in")

plt.minorticks_on()
plt.title(r"log[Mh] predicted from LSTM")

plt.legend(prop={'size': 18})



fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(12,12)

save_path = plot_path + "Mh_log_LSTM_TNG100_difference_histogram" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()
print("Done")

