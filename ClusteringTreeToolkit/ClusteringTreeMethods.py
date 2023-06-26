### Python 3.8.10
import numpy as np                                      # numpy 1.23.3
import pandas as pd                                     # pandas 1.5.0
from matplotlib import pyplot as plt                    # matplotlib 3.6.1
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn import cluster                             # sklearrn 0.0
from scipy.signal import savgol_filter                  # scipy 1.9.1
from sklearn.metrics import calinski_harabasz_score
import time




### colorbar
viridis = cm.get_cmap('RdYlBu_r', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[:5, :] = white
newcmp = ListedColormap(newcolors)



def get_traces_from_datafile(filePath, *, distanceGap=1):
    """
    # Datafile should contain two columns: 1. distance; 2. conductance
    # Traces are cut by sudden change of distance
    # DistanceGap should be smaller than the length of each trace in distance 
    """
    data = pd.read_csv(filePath, header=None, sep='\t')
    data = np.array(data.iloc[:, :])
    distDiff = np.diff(data[:, 0][:])
    traceBegIdx = np.where(np.abs(distDiff) > distanceGap)[0] + 1    # cut by distance gap
    tracesNum = traceBegIdx.shape[0] + 1
    traceBegIdx = traceBegIdx.tolist()
    traceBegIdx.append(data.shape[0])
    traceBegIdx[-1]
    data[traceBegIdx[0]:traceBegIdx[1]]
    traces = []
    traces.append(data[:traceBegIdx[0], :])
    for i in range(tracesNum-1):
        traces.append(data[traceBegIdx[i]:traceBegIdx[i+1], :])
    print("The number of traces:", len(traces))

    return np.array(traces)



def single_trace_standardization(trace, dist_range=(0, 2), dist_space=0.005):
    """
    # Normalize the original trace to a trace with fixed spacing (dist_space):
    # 1. Divide the distance (x-axis) into several fixed spacing bins;
    # 2. The determination of the landing point of the original data follows the principle of left closed and right open;
    # 3. Average data points falling in the same bin; 
    # 4. When there is no data point in the bin, use the last data point from the previous bin
    """
    distBins = np.arange(dist_range[0], dist_range[1], dist_space)
    binsNum = distBins.shape[0]
    dataPointsNum = trace.shape[0]

    curBinCondu = []                # The data points of the current bin
    standardTrace = []              # The conductance of each bin in trace

    binIdx = 0
    dataPointIdx = 0
    while binIdx < binsNum:
        if dataPointIdx >= dataPointsNum:                           # If the original trace has no data point in the last segment, complete the segment with the last data point of the trace
            standardTrace.append(trace[dataPointIdx - 1, 1])
            binIdx += 1
        elif trace[dataPointIdx, 0] < distBins[binIdx]:             # The scanned point falls to the left of the current bin
            dataPointIdx += 1
        elif trace[dataPointIdx, 1]==np.nan or trace[dataPointIdx, 1]==np.inf or trace[dataPointIdx, 1]==-np.inf:       # Check the validity of conductance of the data point
            dataPointIdx += 1
        elif trace[dataPointIdx, 0] >= distBins[binIdx] and trace[dataPointIdx, 0] < distBins[binIdx] + dist_space:     # The scanned point falls in the current bin
            curBinCondu.append(trace[dataPointIdx, 1])
            dataPointIdx += 1
        else:                                                       # The scanned point falls to the right of the current bin
            if len(curBinCondu) == 0:                               # If no data point in the bin, assign it with the current scanned points
                standardTrace.append(trace[dataPointIdx, 1])
            else:
                standardTrace.append(np.array(curBinCondu).mean())
                curBinCondu = []
            binIdx += 1
    standardTrace = np.array(standardTrace)
    return distBins, standardTrace



def traces_standardization(traces, dist_range=(0, 2), dist_space=0.005):
    standard_traces = []
    for i in range(0, traces.shape[0]):
        trace = traces[i]
        dist, condu = single_trace_standardization(trace, dist_range, dist_space)
        standard_trace = np.c_[dist, condu]
        standard_traces.append(standard_trace)

    return np.array(standard_traces)



def getDifference(traces, bins=100, range_=(0, 1), background=-6):
    """
    # Calculate the difference of traces
    # When the conductance drops to the background, set the differential value to a fixed value
    """
    tracesNum = traces.shape[0]
    vectors_ret = []
    gap = (range_[1]-range_[0])/bins
    for i in range(0, tracesNum):
        vector_i_sum = np.zeros(bins)
        vector_i_num = np.zeros(bins)
        need_find_background = True
        background_idx = bins - 2
        for j in range(traces[i].shape[0]):
            if traces[i][j][1]==np.nan or traces[i][j][1]==np.inf or traces[i][j][1]==-np.inf:   # Overbound data may appear in the original dataset, data validation need to be checked
                continue
            if(traces[i][j][0]>=range_[0] and traces[i][j][0]<range_[1]):
                idx = int(((traces[i][j][0]-range_[0])/gap))
                vector_i_sum[idx] += traces[i][j][1]
                vector_i_num[idx] += 1
            if need_find_background and traces[i][j][1]<background:
                background_idx = idx - 1
                need_find_background = False
        vector_i = vector_i_sum / vector_i_num
        vector_i = np.diff(vector_i)
        vector_i = vector_i / gap
        vector_i = np.trunc(vector_i)
        vector_i[background_idx:vector_i.shape[0]-1] = -30
        vectors_ret.append(vector_i)
    vectors_ret = np.array(vectors_ret)

    return vectors_ret 



def horizontal_mapping(traces, bins=300, horizontalRange=(-5.5,-0.3)):
    """
    # Normalized 1D conductance histogram statistics
    """
    NumTraces = len(traces)
    vectors_ret = np.zeros((NumTraces, bins)).astype(np.float32)
    for i in range(NumTraces):
        data_hist_i = np.histogram(traces[i][:,-1], bins, range = horizontalRange, density=False) 
        vectors_ret[i, :] = data_hist_i[0] / np.sum(data_hist_i[0])
        
    return vectors_ret



def vertical_mapping(traces, bins=100, verticalRange=(0,1), dist_space=0.005, smooth_window=50, background=-6):
    """
    # Processing steps：1、standardization；2、smooth；3、calculate difference
    """
    traces_ = []
    for i in range(0, traces.shape[0]):
        trace = traces[i]
        dist, condu = single_trace_standardization(trace, dist_range=verticalRange, dist_space=dist_space)
        condu_smooth = savgol_filter(condu, smooth_window, 2, mode='nearest')
        trace_smooth = np.c_[dist, condu_smooth]
        traces_.append(trace_smooth)
    traces_ = np.array(traces_)
    vectors_ret = getDifference(traces_, bins=bins, range_=verticalRange, background=background)

    return vectors_ret



def clustering_KMeanspp(vectors, maxClustersNum = 14):
    labelsList = []
    for nCl in range(2, maxClustersNum+1):
        kmeans = cluster.KMeans(n_clusters=nCl, random_state=429, init='k-means++')
        kmeans.fit(vectors)
        labels = kmeans.labels_
        labelsList.append(labels)
    labelsList = np.array(labelsList)

    return labelsList



def get_CHI(vectors, labelsList):
    groupsNum = len(labelsList)
    CHI = np.ndarray(groupsNum)
    for i in range(groupsNum):
        CHI[i] = calinski_harabasz_score(vectors, labelsList[i])

    return CHI



def concate_to_one_row(dataCut):
    dataCut = np.vstack(dataCut.tolist())
    dataDist = dataCut[:, 0]
    dataCondu = dataCut[:, -1]

    return dataDist, dataCondu



def get_traces_length(traces, condu_start=-0.3, condu_end=-5.5):
    tracesNum  = traces.shape[0]
    d = np.zeros((tracesNum))
    for i in range(tracesNum):
        DataConduI = traces[i][:, 1]
        index = np.where(DataConduI[:] >= condu_start)[0]
        indexn = np.where(DataConduI[:] <=  condu_end)[0]
        if(index.shape[0] < 1):
            index = [0]
        if indexn.shape[0] < 1:
            indexn = [-1]
        DatadistI = traces[i][:, 0]           
        if(indexn[0] < index[-1]):
            d[i] = 0
        else:
            d[i] = np.abs(DatadistI[index[-1]] - DatadistI[indexn[0]])
    zerosValueIndex = np.where(d <= 0.025)[0]
    d = np.delete(d, zerosValueIndex)
    return d



def get_most_probable_evolution_trace(hist2d):
    intensity = hist2d[0]
    xbins = hist2d[1]
    ybins = hist2d[2]
    xbins = xbins[0:xbins.shape[0]-1]
    ybins = ybins[0:ybins.shape[0]-1]

    x = xbins
    y = []
    for i in range(0,x.shape[0]):
        intensity_i = intensity[i]
        y_i = ybins[np.argmax(intensity_i)]
        y.append(y_i)
    y = np.array(y)

    return x, y



def plot_nodes_2D_histogram(traces, titles, indexes, *, rangeH=((-0.25,1.25),(-7,1)), bins=(100,300), vmax=300, vmin=0, showMPT=False):
    nodesNum = len(titles)
    hists2Ds = []
    plt.figure(figsize=(30,6*((nodesNum-1)//5+1)), dpi=100)
    for i in range(nodesNum):
        xy = traces[indexes[i]]
        x, y = concate_to_one_row(xy)
        x = x.reshape(-1)
        y = y.reshape(-1)
        plt.subplot((nodesNum-1)//5+1, 5, i+1)
        plt.xticks(list([-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]),font='Arial',fontsize='25')
        plt.yticks(list([-7,-6,-5,-4,-3,-2,-1,0,1]),font='Arial',fontsize='25')
        hist2d = plt.hist2d(x, y, bins, range=rangeH, cmap=newcmp, vmax=vmax, vmin=vmin)
        if showMPT:
            x, y = get_most_probable_evolution_trace(hist2d)
            plt.plot(x, y, c='black', linewidth=4.0)
        plt.title("Cluster " + titles[i], font='Arial', fontsize='28')
        plt.gca().spines['left'].set_linewidth(2.0)
        plt.gca().spines['right'].set_linewidth(2.0)
        plt.gca().spines['top'].set_linewidth(2.0)
        plt.gca().spines['bottom'].set_linewidth(2.0)
        hists2Ds.append(hist2d[0])
    plt.tight_layout()
    plt.show()
    return hists2Ds



def plot_nodes_1D_histogram(traces, titles, indexes, *, rangeC=(-6,1), bins=300, yMulFactor=2):
    nodesNum = len(titles)
    hist1Ds = []
    plt.figure(figsize=(30,6*((nodesNum-1)//5+1)), dpi=100)
    for i in range(nodesNum):
        xy = traces[indexes[i]]
        _, y = concate_to_one_row(xy)
        y = y.reshape(-1)
        plt.subplot((nodesNum-1)//5+1, 5, i+1)
        intense, condu, _ = plt.hist(y, bins, range=rangeC)
        plt.xticks(list(range(-7,2)),font='Arial',fontsize='25')
        plt.yticks(font='Arial',fontsize='25')
        plt.xlim(rangeC)
        plt.yticks([])
        plt.ylim(0, np.max(intense)/yMulFactor)
        plt.title("Cluster " + titles[i], font='Arial',fontsize='28')
        plt.gca().spines['left'].set_linewidth(2.0)
        plt.gca().spines['right'].set_linewidth(2.0)
        plt.gca().spines['top'].set_linewidth(2.0)
        plt.gca().spines['bottom'].set_linewidth(2.0)
        condu = condu[0:condu.shape[0]-1]
        hist = np.c_[condu, intense]
        hist1Ds.append(hist)
    plt.tight_layout()
    plt.show()
    return hist1Ds



def plot_nodes_nTraces(traces, titles, indexes, *, showTraceNum=3, space=1.0):
    nodesNum = len(titles)
    plt.figure(figsize=(30,6*((nodesNum-1)//5+1)), dpi=100)
    for i in range(nodesNum):
        xy = traces[indexes[i]]
        TracesNum = xy.shape[0]
        np.random.seed(int(time.time()*1000000%(2**32-1)))
        rand = np.random.randn(showTraceNum) * 100000
        rand = np.abs(rand)
        rand = rand.astype(int)
        rand = rand % TracesNum

        plt.subplot((nodesNum-1)//5+1, 5, i+1)
        for k in range(showTraceNum):
            x_k = xy[rand[k]][:, 0] + k*space
            y_k = xy[rand[k]][:, 1]
            plt.plot(x_k, y_k)
        plt.xticks(font='Arial', fontsize='25')
        plt.yticks(list([-6,-5,-4,-3,-2,-1,0,1]), font='Arial', fontsize='25')
        plt.title("Cluster " + titles[i], font='Arial', fontsize='28')
        plt.ylim(-6,1)
        plt.gca().spines['left'].set_linewidth(2.0)
        plt.gca().spines['right'].set_linewidth(2.0)
        plt.gca().spines['top'].set_linewidth(2.0)
        plt.gca().spines['bottom'].set_linewidth(2.0)
    plt.tight_layout()
    plt.show()



def plot_CHI(CHI):
    plt.figure(figsize=(7,6),dpi=100)
    x = range(2,CHI.shape[0]+2)
    plt.plot(x, CHI,c='#B33F49',linewidth=3.0)
    plt.scatter(x, CHI,c='#B33F49',s=100)
    plt.ylabel('Calinski-Harabasz Score',font='Arial', fontsize='25')
    plt.xlabel('Clusters Number',font='Arial', fontsize='25')
    plt.xticks(font='Arial',fontsize='25')
    plt.yticks(font='Arial',fontsize='25')
    plt.gca().spines['left'].set_linewidth(2.0)
    plt.gca().spines['right'].set_linewidth(2.0)
    plt.gca().spines['top'].set_linewidth(2.0)
    plt.gca().spines['bottom'].set_linewidth(2.0)
    plt.show()



def plot_nodes_plateau_length(traces, titles, indexes, *, bins=150, barWidth=0.005):
    plateauLengthHists  = []
    nodesNum = len(titles)
    plt.figure(figsize=(30,6*((nodesNum-1)//5+1)), dpi=100)
    for i in range(0,nodesNum):
        traces_i = traces[indexes[i]]
        lengths = get_traces_length(traces_i)
        y, x = np.histogram(lengths, bins=bins)
        x = x[0:x.shape[0]-1]
        plt.subplot((nodesNum-1)//5+1, 5, i+1)
        plt.bar(x, y, width=barWidth)
        plt.gca().spines['left'].set_linewidth(2.0)
        plt.gca().spines['right'].set_linewidth(2.0)
        plt.gca().spines['top'].set_linewidth(2.0)
        plt.gca().spines['bottom'].set_linewidth(2.0)
        plt.xticks(font='Arial', fontsize='25')
        plt.yticks(font='Arial', fontsize='25')
        plt.title("Cluster " + titles[i], font='Arial', fontsize='28')
        hist = np.c_[x ,y]
        plateauLengthHists.append(hist)
    plt.tight_layout()
    plt.show()
    return plateauLengthHists





class ClusteringTree():
    """
    # Data structure and interfaces of cluster tree
    """
    def __init__(self, traces, hMapVec, vMapVec, CHIPrimary, titlesChildNodes, indexesChildNodes, CHIsSecondary, titlesLeafNodes, indexesLeafNodes):
        self.traces = traces
        self.hMapVec = hMapVec
        self.vMapVec = vMapVec

        self.CHIPrimary = CHIPrimary
        self.titlesChildNodes = titlesChildNodes
        self.indexesChildNodes = indexesChildNodes
        self.hist2DChildNodes = []
        self.hist1DChildNodes = []
        self.plateauLengthChildNodes = []

        self.CHIsSecondary  = CHIsSecondary
        self.titlesLeafNodes = titlesLeafNodes
        self.indexesLeafNodes = indexesLeafNodes
        self.hist2DLeafNodes = []
        self.hist1DLeafNodes = []
        self.plateauLengthLeafNodes = []


    def plot_CHI_primary(self):
        plot_CHI(self.CHIPrimary)


    def plot_CHIs_secondary(self):
        for i in range(0, len(self.CHIsSecondary)):
            plot_CHI(self.CHIsSecondary[i])


    def plot_child_nodes(self, *, show='all', range2D=((-0.25,1.25),(-7,1)), bins2D=(100,300), vmax=300, vmin=0, range1D=(-6,1), bins1D=300, yMulFactor=2, showMPT=False):
        hist2Ds = plot_nodes_2D_histogram(self.traces, self.titlesChildNodes, self.indexesChildNodes, rangeH=range2D, bins=bins2D, vmax=vmax, vmin=vmin, showMPT=showMPT)
        self.hist2DChildNodes = hist2Ds
        if show=='hist2d':
            return
        hist1Ds = plot_nodes_1D_histogram(self.traces, self.titlesChildNodes, self.indexesChildNodes, rangeC=range1D, bins=bins1D, yMulFactor=yMulFactor)
        self.hist1DChildNodes = hist1Ds
        if show=='hist2d1d':
            return
        plot_nodes_nTraces(self.traces, self.titlesChildNodes, self.indexesChildNodes)


    def plot_leaf_nodes(self, *, show='all', range2D=((-0.25,1.25),(-7,1)), bins2D=(100,300), vmax=300, vmin=0, range1D=(-6,1), bins1D=300, yMulFactor=2, showMPT=False):
        hist2Ds = plot_nodes_2D_histogram(self.traces, self.titlesLeafNodes, self.indexesLeafNodes, rangeH=range2D, bins=bins2D, vmax=vmax, vmin=vmin, showMPT=showMPT)
        self.hist2DLeafNodes = hist2Ds
        if show=='hist2d':
            return
        hist1Ds = plot_nodes_1D_histogram(self.traces, self.titlesLeafNodes, self.indexesLeafNodes, rangeC=range1D, bins=bins1D, yMulFactor=yMulFactor)
        self.hist1DLeafNodes = hist1Ds
        if show=='hist2d1d':
            return
        plot_nodes_nTraces(self.traces, self.titlesLeafNodes, self.indexesLeafNodes)


    def get_child_nodes_plateau_length(self, bins=150, barWidth=0.03):
        plateauLengthHists = plot_nodes_plateau_length(self.traces, self.titlesChildNodes, self.indexesChildNodes, bins=bins, barWidth=barWidth)
        self.plateauLengthChildNodes = plateauLengthHists


    def get_leaf_nodes_plateau_length(self, bins=150, barWidth=0.005):
        plateauLengthHists = plot_nodes_plateau_length(self.traces, self.titlesLeafNodes, self.indexesLeafNodes, bins=bins, barWidth=barWidth)
        self.plateauLengthLeafNodes = plateauLengthHists


    def save_leaf_nodes_plots_data(self, savePath='./PlotsData'):
        hist2dNum = len(self.hist2DLeafNodes)
        hits1dNum = len(self.hist1DLeafNodes)
        if hist2dNum==0:
            print('No hist2D content! Please run plot_child_nodes() first.')
        if hits1dNum==0:
            print('No hist1D content! Please run plot_child_nodes() first.')
        for i in range(0, hist2dNum):
            filePath = savePath + '/Hist2DCluster' + self.titlesLeafNodes[i] + '.txt'
            data = self.hist2DLeafNodes[i]
            np.savetxt(filePath, data, delimiter='\t',fmt='%.3f')
        for i in range(0, hist2dNum):
            filePath = savePath + '/Hist1DCluster' + self.titlesLeafNodes[i] + '.txt'
            data = self.hist1DLeafNodes[i]
            np.savetxt(filePath, data, delimiter='\t',fmt='%.3f')


    def save_leaf_nodes_plateau_length(self, savePath='./PlotsData'):
        nodesNum = len(self.plateauLengthLeafNodes)
        if nodesNum==0:
            print('No content! Please run get_leaf_nodes_plateau_length() first.')
        else:
            for i in range(0, nodesNum):
                filePath = savePath + '/PlateaulengthCluster' + self.titlesLeafNodes[i] + '.txt'
                data = self.plateauLengthLeafNodes[i]
                np.savetxt(filePath, data, delimiter='\t',fmt='%.3f')


    def single_step_horizontal_clustering(self, *, show='all', range2D=((-0.25,1.25),(-7,1)), bins2D=(100,300), vmax=300, vmin=0, range1D=(-6,1), bins1D=300, yMulFactor=2):
        self.plot_child_nodes(show=show, range2D=range2D, bins2D=bins2D, vmax=vmax, vmin=vmin, range1D=range1D, bins1D=bins1D, yMulFactor=yMulFactor)


    def single_step_vertical_clustering(self, *, show='all', range2D=((-0.25,1.25),(-7,1)), bins2D=(100,300), vmax=300, vmin=0, range1D=(-6,1), bins1D=300, yMulFactor=2):
        labelsList = clustering_KMeanspp(self.vMapVec)
        CHI =  get_CHI(self.vMapVec, labelsList)
        ClusterNum = np.argmax(CHI)+2
        labels = labelsList[ClusterNum-2]
        titles = []
        indexes = []
        for i in range(0, ClusterNum):
            index = np.where(labels==i)[0]
            titles.append(str(i+1))
            indexes.append(index)
        
        hist2Ds = plot_nodes_2D_histogram(self.traces, titles, indexes, rangeH=range2D, bins=bins2D, vmax=vmax, vmin=vmin)
        self.hist2DChildNodes = hist2Ds
        if show=='hist2d':
            return
        hist1Ds = plot_nodes_1D_histogram(self.traces, titles, indexes, rangeC=range1D, bins=bins1D, yMulFactor=yMulFactor)
        self.hist1DChildNodes = hist1Ds
        if show=='hist2d1d':
            return
        plot_nodes_nTraces(self.traces, titles, indexes)


def process_traces_clustering_tree(traces, *, horRange=(-5.5,-0.3), horBins=300, verRange=(0,1), verBins=100):
    """
    # Apply the clsuering tree algorithm to the traces
    """
    horizontalVectors = horizontal_mapping(traces, bins=horBins, horizontalRange=horRange)
    try:
        verticalVectors = vertical_mapping(traces, bins=verBins, verticalRange=verRange)
    except:
        print('The value vertical bins too large! Please reduce.')
        return

    CHIPrimary = None
    titlesChildNodes = []
    indexesChildNodes = []
    CHIsSecondary = []
    titlesLeafNodes = []
    indexesLeafNodes = []

    labelsListPrimary = clustering_KMeanspp(horizontalVectors)
    CHIPrimary =  get_CHI(horizontalVectors, labelsListPrimary)
    ClusterNumPrimary = np.argmax(CHIPrimary)+2
    labelsPrimary = labelsListPrimary[ClusterNumPrimary-2]

    for i in range(0, ClusterNumPrimary):
        index = np.where(labelsPrimary==i)[0]
        titlesChildNodes.append(str(i+1))
        indexesChildNodes.append(index)

        labelsListSecondary = clustering_KMeanspp(verticalVectors[index])
        CHISecondary =  get_CHI(verticalVectors[index], labelsListSecondary)
        ClusterNumSecondary = np.argmax(CHISecondary)+2
        labelsSecondary = labelsListSecondary[ClusterNumSecondary-2]
        CHIsSecondary.append(CHISecondary)

        for j in range(0, ClusterNumSecondary):
            index_ = np.where(labelsSecondary==j)[0]
            index_ = index[index_]
            titlesLeafNodes.append(str(i+1)+'-'+str(j+1))
            indexesLeafNodes.append(index_)

    print("Leaf nodes Number: "+str(len(titlesLeafNodes)))

    clusteringTree = ClusteringTree(traces, horizontalVectors, verticalVectors, CHIPrimary, titlesChildNodes, indexesChildNodes, CHIsSecondary, titlesLeafNodes, indexesLeafNodes)

    return clusteringTree



def clustering_tree_processing(filePath=None, *, horRange=(-5.5,-0.3), horBins=300, verRange=(0,1), verBins=100):
    try:
        traces = get_traces_from_datafile(filePath)
    except:
        print('File error! Please check the file or the format of file path.')
        return
    
    clusteringTree = process_traces_clustering_tree(traces, horRange=horRange, horBins=horBins, verRange=verRange, verBins=verBins)

    return clusteringTree


