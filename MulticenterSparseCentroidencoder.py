import numpy as np
import math as mh
import copy
import nonLinearOptimizationAlgorithms as opt
from scipy.special import expit
from utilityScript import standardizeData,calcCentroid
	
class SparseMCE:
	def __init__(self,netConfig):
		self.inputDim = netConfig['inputL']
		self.outputDim = netConfig['outputL']
		self.hLayer = copy.deepcopy(netConfig['hL'])
		if 'outputActivation' in netConfig:
			if netConfig['outputActivation']=='':
				self.outputActivation = 'linear'
			else:
				self.outputActivation = netConfig['outputActivation']
		else:
			self.outputActivation = 'linear'
		self.l1Penalty = netConfig['l1Penalty']
		self.errorFunc = netConfig['errorFunc']
		self.actFunc = netConfig['actFunc']
		self.nItrPre = netConfig['nItrPre']
		self.nItrPost = netConfig['nItrPost']
		self.splWCutoff = 1e-08
		self.maxFeatureCnt = 1000

		self.hLayerTmp = []#I'll use this to pre-train model
		self.nUnits = [self.inputDim]+list(self.hLayer)+[self.outputDim]
		self.nLayers = len(self.nUnits)-1
		self.netW = []
		self.allNetW = []
		self.netWTmp = []#I'll use this to hold the pre-trained weights		
		self.trErrorTrace = []
		self.valErrorTrace = None	
		self.layer_iteration = []
		self.hlNo = 0
		self.layer_error_trace = []		
		self.actFuncTmp =[]
		self.lActFunc = ''
		self.freezeLayerFlag = ''
		self.splWs=[]		
		self.splWsBeforePenalty=[]
		self.splIndices=[]
		self.discardedFList = []
		self.allowedFList = []
		self.oData = None
		self.fW = []
		self.featureList = []
		self.featureWeights = []
		self.classCenters = {}
		
		#Initilize some internal variables for class
		self.hLayerTmp = copy.deepcopy(self.hLayer[1:])
		self.actFuncTmp = copy.deepcopy(self.actFunc[1:])

	def initWeight(self,nUnits):
		nLayers=len(nUnits)-1
		W=[np.random.uniform(-1,1, size=(1+nUnits[i],nUnits[i+1])) / np.sqrt(nUnits[i]) for i in range(nLayers)]
		if self.netWTmp==[]:
			self.netWTmp=[w for w in W]		
		else:
			tmpW=[]
			self.netWTmp[-1] = copy.deepcopy(W[0])
			self.netWTmp.append(W[1])			
		return W

	def flattenD(self,Ws):
		return(np.hstack([W.flat for W in Ws]))

	def unFlattenD(self,Ws):
		#pdb.set_trace()
		sIndex=0
		tmpWs=[]
		Ws=Ws.reshape(1,-1)
		if self.freezeLayerFlag == True:
			for i in range(self.hlNo-1,self.hlNo+1,1):
				d1=np.shape(self.netW[i])[0]
				d2=np.shape(self.netW[i])[1]
				self.netW[i]=(Ws[0,sIndex:sIndex+d1*d2]).reshape(d1,d2)
				sIndex=sIndex+(d1*d2)
		else:
			for i in range(len(self.netW)):
				d1=np.shape(self.netW[i])[0]
				d2=np.shape(self.netW[i])[1]
				self.netW[i]=(Ws[0,sIndex:sIndex+d1*d2]).reshape(d1,d2)
				sIndex=sIndex+(d1*d2)

	def unFlattenDPre(self,Ws):
		#pdb.set_trace()
		sIndex=0
		tmpWs=[]
		Ws=Ws.reshape(1,-1)
		if self.freezeLayerFlag == True:
			for i in range(self.hlNo-1,self.hlNo+1,1):
				d1=np.shape(self.netWTmp[i])[0]
				d2=np.shape(self.netWTmp[i])[1]
				self.netWTmp[i]=(Ws[0,sIndex:sIndex+d1*d2]).reshape(d1,d2)
				sIndex=sIndex+(d1*d2)
		else:
			for i in range(len(self.netWTmp)):
				d1=np.shape(self.netWTmp[i])[0]
				d2=np.shape(self.netWTmp[i])[1]
				self.netWTmp[i]=(Ws[0,sIndex:sIndex+d1*d2]).reshape(d1,d2)
				sIndex=sIndex+(d1*d2)
				
	def returnSPLIndices(self):
		indexList=[]
		for i in range(len(self.splIndices)):
			indexList.append(np.arange(self.splIndices[i][0],self.splIndices[i][1]))
		return np.hstack((indexList))
	
	def returnSPLWs(self,Ws):
		splWs=[]
		Ws=Ws.reshape(1,-1)
		for i in range(len(self.splIndices)):
			splWs.append(Ws[0,self.splIndices[i][0]:self.splIndices[i][1]])
		return np.hstack((splWs)),splWs[0]

	def createOutputAsCentroids(self,data,label):
		centroidLabels=np.unique(label)
		outputData=np.zeros([np.shape(data)[0],np.shape(data)[1]])
		for i in range(len(centroidLabels)):
			indices=np.where(centroidLabels[i]==label)[0]
			tmpData=data[indices,:]
			centroid=np.mean(tmpData,axis=0)
			outputData[indices,]=centroid
		return outputData
		
	def sortDataOnLabel(self,trData,trLabels):
		from operator import itemgetter
		trLabels = trLabels.reshape(-1,1)
		lTrData = np.hstack((trData,trLabels))
		sortedData = np.array(sorted(lTrData, key=itemgetter(-1)))
		return sortedData[:,:-1], sortedData[:,-1].reshape(-1,1)
		
	def createOutputWithMultiCenter(self,trData,trLabels,clusterList,maxItr):
		# use kMean to cluster each class
		from sklearn.cluster import KMeans
		outputData = []
		classes = np.unique(trLabels)
		#pdb.set_trace()
		for i in range(len(classes)):
			#collect the indices of a class
			indices = np.where(trLabels == classes[i])[0]
			#put those samples in a variable call classData
			classData = trData[indices,:]
			#declare a var tmpOutPut with same dim of classData
			tmpOutPut = np.zeros([len(classData),np.shape(classData)[1]])
			#run clustering on the classData
			kmeans = KMeans(n_clusters=clusterList[i], n_init=25, max_iter=maxItr, random_state=0).fit(classData)
			#print('Running kMean for class',classes[i].astype(int),'No of centers:',clusterList[i])
			#in this loop we'll assign the cluster center as output for each cluster member
			for c in range(clusterList[i]):
				patternIndices = np.where(c == kmeans.labels_)[0]
				#now er are using the tmpOutPut, the reason is simple
				tmpOutPut[patternIndices,:] = kmeans.cluster_centers_[c]
				self.classCenters['Cls'+str(int(classes[i]))+'_Cen'+str(int(c))] = kmeans.cluster_centers_[c]
			
			outputData.append(tmpOutPut)
		#pdb.set_trace()
		return np.vstack((outputData))

	def resetNet(self,L):
		self.allNetW.append(copy.deepcopy(self.netW))
		self.netW[0] = np.delete(self.netW[0],L,axis=1)
		self.netW[1] = np.delete(self.netW[1],L,axis=0)
		self.netW[-1] = np.delete(self.netW[-1],L,axis=1)
 
	def feval(self,fName,*args):
		return eval(fName)(*args)
	
	def tanh(self,X):
		return np.tanh(X)

	def sigmoid(self,X):
		return expit(X) #using the sigmoidal of scipy package

	def rect(self,X):
		X[X<=0]=0*X[X<=0]
		return X

	def rectl(self,X):
		X[X<=0]=0.01*X[X<=0]
		return X

	def linear(self,X):
		return X

	def getTrErrorTrace(self):
		return self.trErrorTrace

	def getValErrorTrace(self):
		return self.valErrorTrace
		
	def forwardPassPre(self,D):
		#pdb.set_trace()
		lOut=[D]
		lLength=len(self.netWTmp)
		for j in range(lLength-1):
			d=np.dot(lOut[-1],self.netWTmp[j][1:,:])+self.netWTmp[j][0]#first row in the weight is the bias
			#Take the activation function from the dictionary and apply it
			lOut.append(self.feval('self.'+self.lActFunc,d))
		
		d=np.dot(lOut[-1],self.netWTmp[j+1][1:,:])+self.netWTmp[j+1][0]
		lOut.append(self.feval('self.'+self.outputActivation,d))
		return lOut
		
	def forwardPassSPL(self,D):
		lOut=[D]
		lLength=len(self.netW)
		#pdb.set_trace()
		for j in range(lLength-1):
			if self.actFunc[j]=='SPL':
				lOut.append(lOut[-1]*np.tile(self.netW[j],(np.shape(lOut[-1])[0],1)))#For Sparsity promoting layer
			else:
				d=np.dot(lOut[-1],self.netW[j][1:,:])+self.netW[j][0]#first row in the weight is the bias
				#Take the activation function from the dictionary and apply it
				lOut.append(self.feval('self.'+self.actFunc[j],d))
		d=np.dot(lOut[-1],self.netW[j+1][1:,:])+self.netW[j+1][0]
		lOut.append(self.feval('self.'+self.outputActivation,d))#For output layer
		return lOut

	def backwardPassFreezeLayer(self,error,lO):
		#This will return the partial derivatives for all the layers.
		if self.outputActivation=='sigmoid': #If the output layer is sigmoid then calculate delta 
			deltas=[(lO[-1]*(1-lO[-1]))*error]
		else:# Otherwise assuming that the output layer is linear and delta will be just the error
			deltas=[error]
		#pdb.set_trace()
		for l in range(len(self.netWTmp)-1,0,-1):
			if 'tanh' in self.lActFunc:
			#Activation function: f(x)=(1-exp(-x))/(1+exp(-x))
			#f'(x)=(1-f(x)^2) I'm doing ((1-lO[i]^2))
				delta=(1-lO[l]**2)*(np.dot(deltas[-1],self.netWTmp[l][1:,:].T))
			#Rectifier unit
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0
			elif 'rect' in self.lActFunc:
				derivatives = 1*np.array(lO[l]>0).astype(int)
				delta=derivatives*(np.dot(deltas[-1],self.netWTmp[l][1:,:].T))
			elif 'rectl' in self.lActFunc:
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				derivatives = 0.01*np.array(lO[l]<=0).astype(int)
				derivatives[derivatives==0] = 1
				delta=derivatives*(np.dot(deltas[-1],self.netWTmp[l][1:,:].T))
			elif 'sigmoid' in self.lActFunc:
			#Activation function as f(x)=1/(1+exp(-x))
			#f'(x)=f(x)(1-f(x)) I'm doing (lO[i]*(1-lO[i]))
				delta=(lO[l]*(1-lO[l]))*(np.dot(deltas[-1],self.netWTmp[l][1:,:].T))
			elif 'linear' in self.lActFunc:
				delta=(np.dot(deltas[-1],self.netWTmp[l][1:,:].T))
			else:
				print('Wrong activation function')
			deltas.append(delta)
		deltas.reverse()
		dWs=[]
		#pdb.set_trace()
		for l in range(self.hlNo-1,self.hlNo+1,1):
			#dWs.append(np.vstack((np.dot(lO[l].T,deltas[l]),deltas[l].sum(0))))
			dWs.append(np.vstack((deltas[l].sum(0),np.dot(lO[l].T, deltas[l]))))#The first row is the bias
		return dWs
		
	def backwardPassSPL(self,error,lO):
		#This will return the partial derivatives for all the layers.
		noSample = len(lO[0])
		#pdb.set_trace()
		if self.outputActivation=='sigmoid': #If the output layer is sigmoid then calculate delta 
			deltas=[(lO[-1]*(1-lO[-1]))*error]
		else:# Otherwise assuming that the output layer is linear and delta will be just the error
			deltas=[error]		
		#pdb.set_trace()
		for l in range(len(self.netW)-1,0,-1):
			lActFunc=self.actFunc[l-1]

			if 'tanh' in lActFunc:
			#Activation function: f(x)=(1-exp(-x))/(1+exp(-x))
			#f'(x)=(1-f(x)^2) I'm doing ((1-lO[i]^2))
				delta=(1-lO[l]**2)*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rect' in lActFunc:
			#Rectifier unit function
				derivatives = 1*np.array(lO[l]>0).astype(int)
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rectl' in lActFunc:
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				derivatives = 0.01*np.array(lO[l]<=0).astype(int)
				derivatives[derivatives==0] = 1
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'sigmoid' in lActFunc:
			#Activation function as f(x)=1/(1+exp(-x))
			#f'(x)=f(x)(1-f(x)) I'm doing (lO[i]*(1-lO[i]))
				delta=(lO[l]*(1-lO[l]))*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'linear' in lActFunc:
				delta=(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'SPL' in lActFunc:
				delta=(np.dot(deltas[-1],self.netW[l][1:,:].T))
			else:
				print('Wrong activation function')
			deltas.append(delta)
			
		#splDelta = np.dot(deltas[-1],self.netW[0][1:,:].T)
		#dWs = [(trData*splDelta).sum(0)]
		dWs=[]		
		deltas.reverse()
		#pdb.set_trace()
		for l in range(len(self.netW)-1):
			if self.actFunc[l]=='SPL':
				dWs.append(((lO[l]*deltas[l]).sum(0)).reshape(1,-1))#For sparsity promoting layer
				if self.l1Penalty != None:
					dWs[l] = dWs[l] + self.l1Penalty*np.sign(self.netW[l])
			else:
				dWs.append(np.vstack((deltas[l].sum(0),np.dot(lO[l].T, deltas[l]))))#For hidden layer. First row is the bias
		#pdb.set_trace()
		dWs.append(np.vstack((deltas[l+1].sum(0),np.dot(lO[l+1].T, deltas[l+1]))))#For output layer	
		return dWs
		
	def preTraining(self,iData,oData,verbose=False):
		
		def calcError(cOut):			
			err=(cOut-oData)/(np.shape(oData)[0]*self.outputDim)
			return err

		def costFunc(W):
			self.unFlattenDPre(W)
			lOut=self.forwardPassPre(iData)
			return 0.5 * np.mean((lOut[-1] - oData)**2)
			
		def gradient(W):
			self.unFlattenDPre(W)
			lOut=self.forwardPassPre(iData)
			dWs=self.backwardPassFreezeLayer(calcError(lOut[-1]),lOut)
			return self.flattenD(dWs)
				
		def funcCG(W):
			self.unFlattenDPre(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)
			
		def calcTrErr(W):
			return calcMSE(W,iData,oData)
			
		def calcValErr(W):
			return calcMSE(W,valInput,valOutput)
		
		def calcMSE(W,inputData,outputData):
			##This function will return the RMSE on training data. The error is calculated per data per output dimension
			self.unFlattenDPre(W)
			lOut = self.forwardPassPre(inputData)
			squaredRes = (lOut[-1] - outputData)**2
			rmse = np.sqrt(np.mean(squaredRes))
			return rmse

		self.trSetSize = np.shape(iData)[0]
		self.tmpTrDim = np.shape(iData)[1]
		#Start training for one hidden layer at a time
		#pdb.set_trace()
		iDim = np.shape(iData)[1]
		oDim = np.shape(oData)[1]
		self.freezeLayerFlag = True
		
		for l in range(len(self.hLayerTmp)):
			self.hlNo=l+1
			netLayer = [iDim,self.hLayerTmp[l],oDim]
			W = self.initWeight(netLayer)
			self.lActFunc = self.actFuncTmp[l]
			
			#print('Training layer:',str(netLayer),' with activation function:',self.lActFunc,' No of training data:',len(iData))
			result=opt.scaledconjugategradient(self.flattenD(W), costFunc, gradient, xPrecision=[],fPrecision=[],
					nIterations=self.nItrPre,iterationVariable=self.nItrPre,ftracep=True,verbose=verbose)
			self.unFlattenDPre(result['x'])
			self.layer_error_trace.append(result['ftrace'])
			self.layer_iteration.append(result['nIterations'])
			#print('No of iterations:',self.layer_iteration[-1]-1)
			iDim=self.hLayerTmp[l]
		#if self.counter >0:
		#	pdb.set_trace()
		#Now transfer weights form self.netWTmp to self.netW
		j=0
		for i in range(len(self.actFunc)):
			if self.actFunc[i] == 'SPL':
				#self.netW[i] = np.ones([1,self.tmpTrDim])
				self.netW.append(np.ones([1,self.tmpTrDim]))
			else:
				#self.netW[i] = copy.deepcopy(self.netWTmp[j])
				self.netW.append(copy.deepcopy(self.netWTmp[j]))
				j+=1

		self.netW.append(copy.deepcopy(self.netWTmp[-1]))

		self.freezeLayerFlag=False
		#pdb.set_trace()
		return self

	def postTraining(self,iData,oData,verbose=False):

		def calcError(cOut):
			err=(cOut-oData)/(np.shape(oData)[0]*self.outputDim)			
			return err

		def costFunc(W):
			self.unFlattenD(W)
			lOut = self.forwardPassSPL(iData)
			#pdb.set_trace()
			if self.l1Penalty != None:
				return 0.5 * np.mean((lOut[-1] - oData)**2) + self.l1Penalty*np.sum(np.abs(self.netW[0]))
			else:
				return 0.5 * np.mean((lOut[-1] - oData)**2)

		def gradient(W):
			self.unFlattenD(W)
			lOut=self.forwardPassSPL(iData)
			dWs=self.backwardPassSPL(calcError(lOut[-1]),lOut)
			#pdb.set_trace()
			return self.flattenD(dWs)
				
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)
				
		def calcTrErr(W):			
			return calcMSE(W,iData,oData)
			
		def calcValErr(W):
			return calcMSE(W,valInput,valOutput)
				
		def calcMSE(W,inputData,outputData):
			#This function will return the RMSE on training data. The error is calculated per data per output dimension
			self.unFlattenD(W)
			lOut = self.forwardPassSPL(inputData)
			squaredRes = (lOut[-1] - outputData)**2
			rmse = np.sqrt(np.mean(squaredRes))
			return rmse

		#print('Layerwise pre-training for weight initialization')
		self.preTraining(iData,oData,verbose=verbose)
		self.trSetSize=np.shape(iData)[0]
		#pdb.set_trace()
		#print('Training to initialize wights in SPL. Penalty will not be applied now.')
		tmpL1Penalty = self.l1Penalty
		self.l1Penalty = None
		
		result1 = opt.scaledconjugategradient(self.flattenD(self.netW), costFunc, gradient, xPrecision=[],fPrecision=[],
			nIterations=self.nItrPre,iterationVariable=self.nItrPre,ftracep=True,verbose=verbose)
		
		self.unFlattenD(result1['x'])
		self.splWsBeforePenalty = copy.deepcopy(self.netW[0][0,:])
		
		self.l1Penalty = tmpL1Penalty
		#print('Post training network',self.inputDim,'-->',self.hLayer,'-->',self.outputDim,'with L1 penalty on SPL')
		result2 = opt.scaledconjugategradient(self.flattenD(self.netW), costFunc, gradient, xPrecision=[],fPrecision=[],
				nIterations=self.nItrPost,iterationVariable=self.nItrPost,ftracep=True,verbose=verbose)
		self.unFlattenD(result2['x'])
		self.splWs = copy.deepcopy(self.netW[0][0,:])		
		self.trErrorTrace = result2['ftrace']
		self.iteration = result2['nIterations']-1
		#print('No of SCG iteration:',self.iteration)		
		return self

	def fit(self,trData,trLabels,clusterList=[],maxItrClustering=500,standardizeFlag=True,dynamicPruning=False,verbose=False):
	#def fit(self,trData,trLabels,verbose=True):

		#hold the training sample count
		self.trDataSize = np.shape(trData)[0]
		self.nClass = len(np.unique(trLabels))

		#standardize data
		#pdb.set_trace()
		if standardizeFlag:
			mu,sd,trData = standardizeData(trData)
			self.trMu = mu
			self.trSd = sd

		if clusterList == []: #when clusterList is empty assign 5 centers per class(default)
			clusterList = 2*np.ones(self.nClass).astype(int)
		#sorting the data by labels. Labels are numeric starting from 0
		#This is an important step otherwise wrong center will be assigned to each data after kMean
		trData,trLabels = self.sortDataOnLabel(trData,trLabels)
		#pdb.set_trace()
		target = self.createOutputWithMultiCenter(trData,trLabels,clusterList,maxItrClustering)
		
		#post training	with L1 penalty	
		self.postTraining(trData,target,verbose=verbose)
