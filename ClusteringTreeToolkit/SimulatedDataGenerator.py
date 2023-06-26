import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import time


# colorbar
viridis = cm.get_cmap('RdYlBu_r', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[:5, :] = white
newcmp = ListedColormap(newcolors)


class DataMaker:
    """
    # The class of the generator for simulated single-molecule break junction data.
    # The data form is 'distance-conductance'
    # Conductance plateaus greater than $G_0$ (electrodes contact region) are constructed based on the length of the plateaus
    # Conductance plateaus lower than $G_0$ (molecule signal region) are constructed based on multiple line nodes
    """

    def __init__(self, *, distance_space=0.001, maxCond=1, background=-7, distance_arange=[-0.5,3],
                 ElecPlatLens = pd.DataFrame([[0.15,0.1],[0.1,0.06],[0.2,0.08],[0.1,0.1],[0.3,0.12],[0.3,0.14],[0.3,0.03],[0.3,0.03],[0.3,0.03],[0.3,0.03]],columns=('average_length','length_stddev')),
                 MoleSigNodes = pd.DataFrame([[0,-3,0.01,0.05,0.05],[0.72,-7,0.05,0.05,0.05]],columns=('distance','condutance','distance_stddev','conductance_stddev','Gauss_noise_stddev')),
                 ElecGausnoiseStddev=0.03):

        ###  basic parameters
        self.distance_space = distance_space            # the distance between neighbouring points (equivalent to the reciprocal of data acquisition density)，default 0.001 nm
        self.maxCond = maxCond                          # max value of conductance. default $log(G/G_0)=1$
        self.background = background                    # conductance of back. default $log(G/G_0)=-7$
        self.distance_arange = distance_arange          # The data distance range to construct a trace
        self.distance_arange[0] = 0 if distance_arange[0]>0 else distance_arange[0]     # check validity

        self.ElecPlatLens = ElecPlatLens                # Electrodes_Plateaus_Length，The mean and standard deviation of the length of the conductance plateaus during electrodes contact. The position of conductance are fixed, that is: [0, 0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954, 1](corresponding to 1~10$G_0$).
        self.MoleSigNodes = MoleSigNodes                # Molecule_Signals_Nodes，After the electrodes contact breaking (dist=0,condu=0), trace is assumed be composed by multiple lines, each lines contains five parameters: length on distance; the condutance of the terminal point、the standard deviation of length; the standard deviation of the condutance of the terminal point; the standard deviation of Gauss noise。default: a trace with a snapback to $log(G/G0)=-3$ and an attenuation constant $β=50/9 nm^{-1}$。

        self.ElecGausnoiseStddev = ElecGausnoiseStddev  # Standard deviation of Gaussian noise of the plateaus during electrodes contact

        ###  default parameters
        self.ElecPlatCond = [0, 0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954, 1]      # the position of electrodes contact plateaus

        ###  generated traces
        self.trace_single = pd.DataFrame(columns=['distance','conductance'])        # the trace latest generated
        self.trace_assemble = pd.DataFrame(columns=['distance','conductance'])      # all generated traces 


    def make_single_trace(self):
        """
        # generate a single trace 
        """
        ### prepare distance array
        distances = np.arange(self.distance_arange[0], self.distance_arange[1], self.distance_space)
        self.trace_single['distance'] = distances

        ##### prepare conductance array of electrodes contact region
        elec_cont_data = np.array([])
        elec_cont_data_len = int((0-self.distance_arange[0])//self.distance_space)           # the data length of electrodes contact region, the data point where $distance=0$ is not contains
        elec_cont_data_len_remain = elec_cont_data_len

        for idx,cond in enumerate(self.ElecPlatCond):
            if elec_cont_data_len_remain <= 0:
                break
            np.random.seed(int(time.time()*1000000%(2**32-1)))
            plateauLen = np.random.normal(loc=self.ElecPlatLens.loc[idx,'average_length'], scale=self.ElecPlatLens.loc[idx,'length_stddev'])    # obtain the actual length value of the segment
            plateauLen = 0 if plateauLen<0 else plateauLen                                                                                      # ensure that the length of the segment is equal to or greater than 0
            data_len = int(plateauLen//self.distance_space)
            if data_len==0:
                continue
            np.random.seed(int(time.time()*1000000%(2**32-1)))
            part = np.random.normal(loc=cond, scale=self.ElecGausnoiseStddev, size=data_len)
            elec_cont_data = np.concatenate((elec_cont_data,part))
            elec_cont_data_len_remain -= data_len

        if elec_cont_data_len_remain > 0:
            np.random.seed(int(time.time()*1000000%(2**32-1)))
            part = np.random.normal(loc=self.maxCond, scale=self.ElecGausnoiseStddev, size=100-int(elec_cont_data_len_remain))
            elec_cont_data = np.concatenate((elec_cont_data, part))

        elec_cont_data = elec_cont_data[0:elec_cont_data_len+1]     # splitting
        elec_cont_data = np.flip(elec_cont_data)                    # reverse order arrangement

        ##### prepare conductance array of molecule junction formed region
        mole_cont_data = np.array([])
        mole_cont_data_len = int((self.distance_arange[1] - 0) // self.distance_space)          # the data length of molecule junction formed region, the data point where $distance=0$ is contains

        lastNodePos = [0, 0]                                                                    # the position of the terminal points of the next segment
        for index, node in self.MoleSigNodes.iterrows():
            np.random.seed(int(time.time()*1000000%(2**32-1)))
            part_len = np.random.normal(loc=node['distance'], scale=node['distance_stddev'])    # get the length value of the segment on distance
            part_len = 0 if part_len<0 else part_len                                            # ensure the value is equal to or greater than 0
            np.random.seed(int(time.time()*1000000%(2**32-1)))
            terminal_cond = np.random.normal(loc=node['condutance'], scale=node['conductance_stddev'])     # get the conductance value of the terminal point
            data_len = int(part_len // self.distance_space)
            if data_len>0:
                part_origin = np.linspace(lastNodePos[1],terminal_cond,data_len)
                np.random.seed(int(time.time()*1000000%(2**32-1)))
                part_random = np.random.normal(loc=0, scale=node['Gauss_noise_stddev'], size=data_len)
                part = part_origin + part_random
                mole_cont_data = np.concatenate((mole_cont_data,part))
            lastNodePos = [lastNodePos[0]+part_len,terminal_cond]

        if mole_cont_data.shape[0]<mole_cont_data_len:                                         # If the generated data length is not enough, complete the tail of trace by background conductance. If enough, cut the trace.
            np.random.seed(int(time.time()*1000000%(2**32-1)))
            part = np.random.normal(loc=self.background, scale=self.ElecGausnoiseStddev, size=mole_cont_data_len-mole_cont_data.shape[0]+1)
            mole_cont_data = np.concatenate((mole_cont_data, part))
        else:
            mole_cont_data = mole_cont_data[0:mole_cont_data_len+1]

        ##### merge the two parts
        cont_data = np.concatenate((elec_cont_data,mole_cont_data))
        self.trace_single['conductance'] = cont_data

        return self.trace_single


    def make_traces(self, TraceNum=2):
        for i in range(TraceNum):
            try:
                self.make_single_trace()
                self.trace_assemble = pd.concat((self.trace_assemble, self.trace_single))
            except:     # Due to random numbers causing accidental construction errors, nested construction is used here
                try:
                    self.make_single_trace()
                    self.trace_assemble = pd.concat((self.trace_assemble, self.trace_single))
                except:
                    try:
                        self.make_single_trace()
                        self.trace_assemble = pd.concat((self.trace_assemble, self.trace_single))
                    except:
                        try:
                            self.make_single_trace()
                            self.trace_assemble = pd.concat((self.trace_assemble, self.trace_single))
                        except:
                            print("unexpected error! Please try again!")       


    def plot_current_single_trace(self, *, x_range=[-0.5, 3], y_range=[-7,1]):
        """ 
        # Plot single generated trace
        """
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            print("Plot x_range or y_range inputs error!")
            return
        x = self.trace_single['distance'].values
        y = self.trace_single['conductance'].values
        plt.plot(x,y)
        plt.xlabel("Distance / nm")
        plt.ylabel("Conductance / log(G/G_0)")
        plt.title("Single Trace")
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.show()


    def plot_trace_assemble_heatmap(self, *, x_range=[-0.5, 3], y_range=[-7,1], x_bin_len=0.01, y_bin_len=0.05):
        """ 
        # Plot 2D histogram of the generated traces 
        """
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            print("Plot x_range or y_range inputs error!")
            return
        
        plt.figure(figsize=(7.5,6), dpi=100)
        xs = self.trace_assemble['distance'].values
        ys = self.trace_assemble['conductance'].values
        plt.hist2d(xs, ys, bins=(np.arange(x_range[0], x_range[1], x_bin_len), np.arange(y_range[0], y_range[1], y_bin_len)), cmap=newcmp)
        plt.xlabel("Distance / nm")
        plt.ylabel("Conductance / log(G/G_0)")
        plt.title("Traces 2D Heatmap")
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.show()


    def save_trace_assemble(self, path, mode='a'):
        """
        # save the traces data
        """
        self.trace_assemble.to_csv(path, sep='\t', header=False, index=False, float_format='%.3f', mode=mode)

