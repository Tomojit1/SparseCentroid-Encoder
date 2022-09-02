import numpy as np
from utilityScript import *
from simpleANNClassifierPyTorch import *
from SparseCentroidencoder import SparseCE
import pickle
from sklearn.metrics import accuracy_score

def load_ALLAML_Data(dataSetName,partition):
	
	trnSet,tstSet = getApplicationData(dataSetName,partition)
	return trnSet,tstSet

def runSCE(l1Penalty,trData,trLabels):
	
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
	model = SparseCE(dict2)
	model.fit(trData,trLabels)
	featureList,featuresW = returnImpFeaturesElbow(model.splWs)

	print('SCE ran complete. L1 Penalty',l1Penalty)
	
	return featureList
	
def classifyALLAMLData(trnSet,tstSet,featureSet,fCntList,gpuId,partition):
	
	accuracyList = []
	for feaCnt in fCntList:
		trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
		tstData,tstLabels = tstSet[:,:-1],tstSet[:,-1]
		fea = featureSet[:feaCnt]
	
		#use the selected features
		trData,tstData = trData[:,fea],tstData[:,fea]
		nClass = len(np.unique(trLabels))
		allACC = []
		for i in range(10):
			ann = NeuralNet(trData.shape[1], [500] , nClass)
			ann.fit(trData,trLabels,standardizeFlag=True,batchSize=200,optimizationFunc='Adam',learningRate=0.001, numEpochs=25,cudaDeviceId=gpuId)
			ann = ann.to('cpu')
			tstPredProb,tstPredLabel = ann.predict(tstData)
			accuracy = 100 * accuracy_score(tstLabels.flatten(), tstPredLabel)
			allACC.append(accuracy)
		allACC = np.hstack((allACC))
		accuracyList.append(np.round(np.mean(allACC),2))
		print('Repetition:',partition,'Accuracy using',trData.shape[1],'features:',np.round(np.mean(allACC),2))
	return accuracyList


if __name__ == "__main__":
	
	#set parameters
	l1Penalty = 0.0002
	fCntList = [10,50]
	gpuId = 0

	topTenFeaturesAcc,topFiftyFeaturesAcc = [],[]
	for pp in range(20):
		#load training data data
		trnSet,tstSet = load_ALLAML_Data('ALLAML',pp)
		trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
		
		#run SCE on training data for feature selection
		feaList = runSCE(l1Penalty,trData,trLabels)

		#using the selected features run classification
		accuracyList = classifyALLAMLData(trnSet,tstSet,feaList,fCntList,gpuId,pp)
		topTenFeaturesAcc.append(accuracyList[0])
		topFiftyFeaturesAcc.append(accuracyList[1])
	print('Mean accuracy using top 10 features over 20 data partitions',np.round(np.mean(topTenFeaturesAcc),2))
	print('Mean accuracy using top 50 features over 20 data partitions',np.round(np.mean(topFiftyFeaturesAcc),2))


