import sys, os
sys.path.insert(0, '..')
from lib import models, graph, coarsening, utils
import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
## It works with tensorflow V1
import tensorflow.compat.v1 as tf
import numpy as np
import time
import h5py
import scipy.io as sio
import pathlib
import json
import importlib
importlib.reload(utils)
importlib.reload(models)

#matplotlib inline.
## My_Usage function
def usage():
    print('You should pass the method to use:\n\
        \tPPIS:\tFor PPI + singletons\n\
        \tPPI:\tFor only PPI\n\
        \tCOEX:\tFor coexpression\n\
        \tCOEXS:\tFor coexpression + singletons')
    exit(-1)

flags = tf.app.flags
FLAGS = flags.FLAGS

## My_code_args
if len(sys.argv) < 2: usage()
    
## /My_code_args

# Graphs.
flags.DEFINE_integer('number_edges', 1, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 1, 'Number of coarsened graphs.')
t_start = time.process_time()

## My_code_mats
if 'PPIS' in sys.argv[1]: 
    ## PPI + singletons
    print('Is ppis')
    gene_data='./input_data/Block_PPIA.mat' ## Gene data
    adjacency_mat_data='./input_data/Adj_Filtered_List_0Con.mat' ## Adjacency matrix
    in_shape=(4444,4444) ## Shape for sparse matrix
elif 'PPI' in sys.argv[1]:
    ## Only PPI
    print('Is ppi')
    gene_data='./input_data/Block_PPI1.mat'
    adjacency_mat_data= './input_data/Adj_Filtered_List_0Con.mat'
    in_shape=(4444,4444)
elif 'COEXS' in sys.argv[1]:
    ## Coexpression + singletons
    print('Is coex+ singleton')
    gene_data='./input_data/Block_6P.mat'
    adjacency_mat_data= './input_data/Adj_Spearman_6P.mat'
    in_shape=(3866,3866)
elif 'COEX' in sys.argv[1]:
    ## Only coexpression
    print('Is coexpression only')
    gene_data='./input_data/Block_6PA.mat'
    adjacency_mat_data= './input_data/Adj_Spearman_6P.mat'
    in_shape=(3866,3866)
else: usage() ## Input error


# Directories.
flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')
## USELESS now
# # #For PPI and PPI-singleton model change file location
# # test = sio.loadmat('./input_data/Adj_Filtered_List_0Con.mat')
# # # for Correlaton model change file location
# # #test = sio.loadmat('C:/Users/RJ\Desktop/exp_fpkm_pancan/processed/CoExpression/Adj_Data/Adj_Spearman_6P.mat')

test = sio.loadmat(adjacency_mat_data)      ## Load Matlab file with adjacency matrix
row = test['row'].astype(np.float32)        ## Loads data labeled as row from adjacency matrix
col = test['col'].astype(np.float32)        ## Loads data labeled as col from adjacency matrix
value = test['value'].astype(np.float32)    ## Loads data labeled as value from adjacency matrix
M, k = row.shape
row = np.array(row)                         ## Creates numpy array shape (1, xxxxx)
row = row.reshape(k)                        ## Creates List from the position [0][0]
row = row.ravel()                           ## Doesn't flatten anything
col = np.array(col)                         ## Creates numpy array shape (1, xxxxx)
col = col.reshape(k)                        ## Creates List from the position [0][0]
col = col.ravel()                           ## Creates List from the position [0][0]
value = np.array(value)                     ## Creates numpy array shape (1, xxxxx)
value = value.reshape(k)                    ## Creates List from the position [0][0]
value = value.ravel()                       ## Doesn't flatten anything

## Source: https://scipy-lectures.org/advanced/scipy_sparse/coo_matrix.html
## ++ https://en.wikipedia.org/wiki/Sparse_matrix
A = scipy.sparse.coo_matrix((value, (row, col)),\
    shape = in_shape)                       ## A = Adjacency matrix with size in_shape
                                            ## in_shape selected from graph type
## graphs = 2 graphs. In position [0] the original, in posotion [1] the coarsened one using metis algorithm
## perm= parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
##          which indicate the parents in the coarser graph[i+1]
## Metis info: https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-895-theory-of-parallel-systems-sma-5509-fall-2003/projects/kasheff.pdf
graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=True)

## Graph library
## For each graph creates the laplacian matrix
## laplacian matrix= degree matrix + adjacency matrix
## info laplacian: https://en.wikipedia.org/wiki/Laplacian_matrix
L = [graph.laplacian(A, normalized=True,renormalized=True) for A in graphs]
## Code added to save the graphs in L in pajek format
# import networkx as nx
# i=0
# for l in L:
#     G= nx.from_scipy_sparse_matrix(l)
#     nx.write_pajek(G, "./PPI_{}.net".format(i))
#     i+=1
    
## Deletes useless variables
del test
del A
del row
del col
del value
## Modified to autoselect test mat 
#Data = sio.loadmat('./input_data/Block_PPIA.mat')
## Load Matlab file with 5 data blocks
Data = sio.loadmat(gene_data) ## Load Matlab object with gene data
## Divide blocks from dictionary of size= 5
Data1 = Data['Block'][0,0]
Data2 = Data['Block'][0,1]
Data3 = Data['Block'][0,2]
Data4 = Data['Block'][0,3]
Data5 = Data['Block'][0,4]
## D= Matrices with doubles (depending on the selected computation loads one or other)
D1= Data1['D'].astype(np.float32)
D2= Data2['D'].astype(np.float32)
D3= Data3['D'].astype(np.float32)
D4= Data4['D'].astype(np.float32)
D5= Data5['D'].astype(np.float32)
## L= doubles array with resulting class labels
L1= Data1['L'].astype(np.float32)
L2= Data2['L'].astype(np.float32)
L3= Data3['L'].astype(np.float32)
L4= Data4['L'].astype(np.float32)
L5= Data5['L'].astype(np.float32)

## Final own random code
allData= np.transpose(np.hstack((D1,D2,D3,D4,D5)))
allLabel = (np.vstack((L1,L2,L3,L4,L5)))
## Set the % of data used for Train.
## The rest will be divided in two for validation and test
TrainPercentage= 70
TrainPercentage=1-(TrainPercentage/100)
indexes= np.random.choice(np.arange(allData.shape[0]), int(allData.shape[0]*TrainPercentage), replace=False)
indexes.sort()
Test_Data=[]
Train_Data=[]
Test_Label=[]
Train_Label=[]
Val_Data=[]
Val_Label=[]
i=-1
for ind, e in enumerate(allData):
    
    if ind in indexes: 
        i+=1
        if i%2==0:
            Test_Data.append(allData[ind, :])
            Test_Label.append(allLabel[ind, :])
        else:
            Val_Data.append(allData[ind, :])
            Val_Label.append(allLabel[ind, :])
    else: 
        Train_Data.append(allData[ind, :])
        Train_Label.append(allLabel[ind, :])

Train_Data=np.array(Train_Data)
Train_Label=np.array(Train_Label)

Test_Data=np.array(Test_Data)
Test_Label=np.array(Test_Label)

Val_Data=np.array(Val_Data)
Val_Label=np.array(Val_Label)

Train_Label = Train_Label.ravel()
Val_Label = Val_Label.ravel()
Test_Label = Test_Label.ravel()

Train_Data = coarsening.perm_data(Train_Data, perm)
Val_Data = coarsening.perm_data(Val_Data, perm)
Test_Data = coarsening.perm_data(Test_Data, perm)
## 33 cancer types + 1 normal
C = 34  # number of classes
## Creates an instance of class model_perf
model_perf = utils.model_perf() #utilsNo.model_perf()
## Creates a dict for parameters
common = {}
common['dir_name']       = 'PPI/'
common['num_epochs']     = 20
common['batch_size']     = 200
common['decay_steps']    = 17.7 # * common['num_epochs'] since not used use as in momentum 
common['eval_frequency'] = 10 * common['num_epochs']
common['brelu']          = 'b1relu'
common['pool']           = 'apool1'
common['regularization'] = 0
common['dropout']        = 1
## Adjust learning rate as paper indicates, 0.001 no singletons, 0.005 singletons
if 'S' in sys.argv[1]: common['learning_rate']  = .005
else: common['learning_rate']  = .001
common['decay_rate']     = 0.95
common['momentum']       = 0
common['F']              = [1]
common['K']              = [1]
common['p']              = [2]
common['M']              = [1024,C]

name = 'Run1'
## Copy common to params, adds filterparameter and rewrites brelu and dir_name
params = common.copy()
params['dir_name'] += name
# params['filter'] = 'chebyshev5'
params['filter'] = 'chebyshev2'
params['brelu'] = 'b1relu'
## To avoid compute the gene perturbation replace import utils by utilsNO
model_perf.test(models.cgcnn(L, **params), name, params, Train_Data, Train_Label, Val_Data, Val_Label, Test_Data, Test_Label)

model_perf.show()
## Grid search
if False:
    grid_params = {}
    data = (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    utilsNo.grid_search(params, grid_params, *data, model=lambda x: models.cgcnn(L,**x))
print(sys.argv[1])