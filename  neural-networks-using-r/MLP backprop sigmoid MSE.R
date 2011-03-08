# Standard Backpropagation FeedForward Neural Network
# 
# Author: Gustavo (lgbpinho153@yahoo.com.br)
###############################################################################

#
# Samples 
##########

I<-rbind(runif(100),runif(100),runif(100))
T<-matrix(10+2*I[1,]+3*I[2,]+5*I[3,],1,100)

#
# Parameter
#############
N<-dim(I)[2] # number of samples
NumInputs<-dim(I)[1]
NumHidden<-5
NumOutputs<-dim(T)[1]
WeightsHI<-matrix(runif(NumHidden*(NumInputs+1)),NumHidden,(NumInputs+1))
WeightsOH<-matrix(runif(NumOutputs*(NumHidden+1)),NumOutputs,NumHidden+1)
Lr<-0.01
NumEpochs<-100

#
# Pre-process
##############
Inputs<-matrix(0,NumInputs,N)
Targets<-matrix(0,NumOutputs,N)
for (i in 1:NumInputs){
	Inputs[i,]<-0.99*(I[i,]-min(I[i,]))/(max(I[,])-min(I[i,]))+0.001	
}
for (i in 1:NumOutputs){
	Targets[i,]<-0.99*(T[i,]-min(T[i,]))/(max(T[,])-min(T[i,]))+0.001	
}

#
# Sigmoid
###########
phi<-function(v) 1/(1+exp(-v))
phidash<-function(v) phi(v)*(1-phi(v))

#
# Training
############
MSE<-rep(0,NumEpochs)
for (epoch in 1:NumEpochs){
	for(s in 1:N){ # adjusts are made after each sample
		# forward pass
		HiddenInputs<-c(1,Inputs[,s])
		HiddenInducedFields<-WeightsHI%*%HiddenInputs
		HiddenOutputs<-phi(HiddenInducedFields)
		OutputInputs<-c(1,HiddenOutputs)
		OutputInducedFields<-WeightsOH%*%OutputInputs
		Outputs<-phi(OutputInducedFields)
		
		#backpropagation
		DeltaO<-(Targets[,k]-Outputs)*Outputs*(1-Outputs) # local gradient for the output neurons
		DeltaH<-(HiddenOutputs*(1-HiddenOutputs))*WeightsOH[,-1]%*%DeltaO # local gradient for the hidden neurons		
		#ajeitar a linha acima para mais de uma saida
		
		DeltaWOH<-Lr*kronecker(DeltaO,t(OutputInputs))
		DeltaWHI<-Lr*kronecker(DeltaH,t(HiddenInputs))
		
		WeightsOH<-WeightsOH+DeltaWOH
		WeightsHI<-WeightsHI+DeltaWHI
	}
	
	# simulating outputs
	HiddenInputs<-rbind(1,Inputs)
	HiddenInducedFields<-WeightsHI%*%HiddenInputs
	HiddenOutputs<-phi(HiddenInducedFields)
	OutputInputs<-rbind(1,HiddenOutputs)
	OutputInducedFields<-WeightsOH%*%OutputInputs
	Outputs<-phi(OutputInducedFields)
	
	MSE[epoch]<-(Targets-Outputs)%*%t(Targets-Outputs)/(2*N)  ### ajeitar isso depois
		
}

plot(MSE,type='l')