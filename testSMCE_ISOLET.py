import numpy as np
from utilityScript import *
from simpleANNClassifierPyTorch import *
from MulticenterSparseCentroidencoder import SparseMCE
import pickle
from sklearn.metrics import accuracy_score

def load_Isolet_Data(dataSetName,partition=None):
	
	trnSet,tstSet = getApplicationData(dataSetName,partition)
	
	return trnSet,tstSet

def runSCE(l1Penalty,trData,trLabels,centersPerClass):
	nClass = len(np.unique(trLabels))
	centerList = centersPerClass*(np.ones(nClass).astype(int))
	dict2 = {}
	dict2['inputL'] = np.shape(trData)[1]
	dict2['outputL'] = np.shape(trData)[1]
	dict2['hL'] = [np.shape(trData)[1],100]
	dict2['actFunc'] = ['SPL','tanh']
	dict2['outputActivation'] = 'linear'
	dict2['l1Penalty'] = l1Penalty
	dict2['nItrPre'] = 10
	dict2['nItrPost'] = 50
	dict2['errorFunc']='MSE'

	#initiate an object of the model and call it's training method 
	model = SparseMCE(dict2)
	model.fit(trData,trLabels,centerList)
	featureList,featuresW = returnImpFeaturesElbow(model.splWs)
	print('SCE ran complete. L1 Penalty',l1Penalty)

	return featureList
	
def classifyIsoletData(trnSet,tstSet,featureSet,fCntList,gpuId,pp):
	
	accuracyList = []
	for feaCnt in fCntList:
		trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
		tstData,tstLabels = tstSet[:,:-1],tstSet[:,-1]
		
		#use the selected features
		fea = featureSet[:feaCnt]
		trData,tstData = trData[:,fea],tstData[:,fea]
		nClass = len(np.unique(trLabels))
		allACC = []
		for i in range(10):
			ann = NeuralNet(trData.shape[1], [1000] , nClass)
			ann.fit(trData,trLabels,standardizeFlag=True,batchSize=200,optimizationFunc='Adam',learningRate=0.001, numEpochs=200,cudaDeviceId=gpuId)
			ann = ann.to('cpu')
			tstPredProb,tstPredLabel = ann.predict(tstData)
			accuracy = 100 * accuracy_score(tstLabels.flatten(), tstPredLabel)
			allACC.append(accuracy)
			
		allACC = np.hstack((allACC))
		accuracyList.append(np.round(np.mean(allACC),2))
		print('Repetition:',pp+1,'Accuracy using',trData.shape[1],'of features:',np.round(np.mean(allACC),2))
	return accuracyList


if __name__ == "__main__":
	
	#set parameters
	l1Penalty = 0.001
	centersPerClass = 5
	fCntList = [50]
	gpuId = 0

	topFiftyFeaturesAcc = []
	for pp in range(20):
		#load training data data
		trnSet,tstSet = load_Isolet_Data('ISOLET')
		trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
		
		#run SCE on training data for feature selection
		feaList = runSCE(l1Penalty,trData,trLabels,centersPerClass)

		#using the selected features run classification
		accuracyList = classifyIsoletData(trnSet,tstSet,feaList,fCntList,gpuId,pp)
		topFiftyFeaturesAcc.append(accuracyList[0])
	print('Mean accuracy using top 50 features over 20 run',np.round(np.mean(topFiftyFeaturesAcc),2))


