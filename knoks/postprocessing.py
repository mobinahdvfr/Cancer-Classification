import numpy as np
import sys
import matplotlib as plt
import matplotlib.pyplot as plt; plt.rcdefaults()

## Load data from CSV file
print('Loading up data...', end='\r')
upData= np.genfromtxt(sys.argv[1]+'Up.csv', delimiter=',').astype(np.uint8)
print('Loading down data...', end='\r')
downData= np.genfromtxt(sys.argv[1]+'Down.csv', delimiter=',').astype(np.uint8)
## Take first row as ground truth
groundTruth=upData[0,:]
## Delete first row for up and down data
upData= np.delete(upData, 0, 0)
downData= np.delete(downData, 0, 0)

## Class error counter
countgenes= np.zeros(34, dtype=int)
print('Computing tumour affectation...', end='\r')
## Go trough all results in up and down
for i in range(upData.shape[1]):
    ## Get all i column from loaded data
    tmpUp= upData[:, i]
    tmpDown= downData[:, i]
    ## uniqueUp/Down contains all the classes in the column
    ## CountUp/Down contains the repetitions for each class 
    uniqueUp, countsUp = np.unique(tmpUp, return_counts=True)
    uniqueDown, countsDown = np.unique(tmpDown, return_counts=True)
    ## Create dict from results
    u= dict(zip(uniqueUp, countsUp))
    d= dict(zip(uniqueDown, countsDown))
    ## For each key in dict...
    for key in d:
        ## Check if key is different in up and down dict due to gene alteration
        if key != groundTruth[i] or not (key in u.keys()): 
            ## Add one error to the temp result
            ## Count one error for a tumour class (groundTruth[i] contains the right class)
            countgenes[groundTruth[i]]+=1
        elif key in u.keys() and u[key]!= groundTruth[i]:
            countgenes[groundTruth[i]]+=1

## Prepare the strings for the plot
tumours=['NT','ACC','BLCA','BRCA','CESC','CHOL','COAD','DLBC','ESCA','GBM','HNSC','KICH','KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','MESO','OV','PAAD','PCPG','PRAD','READ','SARC','SKCM','STAD','TGCT','THCA','THYM','UCEC','UCS','UVM']
namesStr=''
numbersStr='\n'
print('Saving tumour data...', end='\r')
for t, c in zip(tumours, countgenes):
    namesStr += '{},'.format(t)
    numbersStr += '{},'.format(c)
## Remove last comma in both arrays
namesStr=namesStr[: -1]
numbersStr= numbersStr[: -1]
## Save data in CSV format
with open(sys.argv[1]+'_tumours_afected.csv', 'w') as f:
    f.write(namesStr)
    f.write(numbersStr)
    f.close()
## Plot the data into file using MatPlotLib
plt.bar(np.arange(len(tumours)), countgenes, align='center', alpha=0.5)
plt.xticks(np.arange(len(tumours)), tuple(tumours))
plt.ylabel('number affecting genes')
plt.xlabel('Tumours')
plt.title('Class affecting genes {}'.format(sys.argv[1]))
plt.xticks(rotation=45, fontsize=6)
plt.savefig(sys.argv[1]+'_tumours_afected.png', dpi=300)
plt.clf()

print('Computing gene affectation...', end='\r')

## Count the number of errors generated for each one of the genes
resList= np.zeros((upData.shape[0]), dtype=int)
## For each gene modification
for i in range(upData.shape[0]):
    err=0
    ## For each prediction in [i] gene...
    for j in range(upData.shape[1]):
        ## If up or down prediction is different, add one error
        if upData[i][j] != groundTruth[j] or downData[i][j] != groundTruth[j]:
            err+=1
    ## Set the number of errors in gene position
    resList[i]=err
print('Saving gene data...', end='\r')
## Save data in CSV format
with open(sys.argv[1]+'_genes afecting.csv', 'w') as f:
    f.write(','.join(str(e) for e in resList.tolist()))
    f.close()

## Plot the data into file using MatPlotLib
plt.plot(np.arange(len(resList)), [e for e in resList], linewidth=0.75)
plt.ylabel('Results affected')
plt.xlabel('Genes')
plt.title('Affectation by gene {}'.format(sys.argv[1]))
plt.savefig(sys.argv[1]+'_genes_afecting.png', dpi=600)