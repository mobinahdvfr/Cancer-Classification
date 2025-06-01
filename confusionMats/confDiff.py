import numpy as np
import sys
import matplotlib as plt
import matplotlib.pyplot as plt
import sklearn.metrics as met


data1= np.genfromtxt(sys.argv[1], delimiter=',')
data2= np.genfromtxt(sys.argv[2], delimiter=',')

result= data1 - data2

tumours=['NT','ACC','BLCA','BRCA','CESC','CHOL','COAD','DLBC','ESCA','GBM','HNSC','KICH','KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','MESO','OV','PAAD','PCPG','PRAD','READ','SARC','SKCM','STAD','TGCT','THCA','THYM','UCEC','UCS','UVM']
fig, ax = plt.subplots()
im = ax.imshow(result, aspect='auto', interpolation='nearest', cmap='Wistia') #cmap
for i in range(len(tumours)):
    for j in range(len(tumours)): 
        tmpstr='.'
        if result[i][j] != 0: tmpstr=str(int(round(result[i, j],0)))
        if i != j: ax.text(j, i, tmpstr, ha="center", va="center", color="b", size= 'xx-small')
        else: ax.text(j, i, tmpstr, ha="center", va="center", color="r", size= 'xx-small')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_xticks(np.arange(len(tumours)))
ax.set_yticks(np.arange(len(tumours)))
ax.set_xticklabels(tumours)
ax.set_yticklabels(tumours)
for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(6)
ax.set_aspect(0.65)
fig.tight_layout()
plt.title('Confusion difference {}'.format(sys.argv[3]))
plt.xlabel('True class', fontsize=5)
plt.ylabel('Predicted class', fontsize=5)
plt.savefig('./{}_difference.png'.format(sys.argv[3]), dpi=400)