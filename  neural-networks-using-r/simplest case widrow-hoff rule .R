# Implements the delta rule learning or Widrow-Hoff rule (Widrow and Hoff, 1960) 
# as see in Haykin (1994, p.73) for one single neuron in the output layer and
# no hidden layer. Batch training. This is the linear regression model.
# Author: Gustavo (lgbpinho153@yahoo.com.br)
###############################################################################

### Simple example data
In<-cbind(rnorm(100,0,1),rnorm(100,0,1)) # One case per row, one input per column
Target<-10+2*In[,1]+3*In[,2]

### Parameters
NumInputs<-dim(In)[2] # Number of input nodes
NumOutputs<-dim(In)[2] # Number of output nodes
Weights<-rep(0,NumInputs+1) # Bias and weights vector
Lr<-0.05 # Learning rate
Delta<-rep(0,NumInputs+1) # Weight adjustment vector
NumEpochs<-1000

### Estimation
NeuronIn<-cbind(rep(1,100),In)
for (j in 1:NumEpochs){
	NeuronSum<-NeuronIn%*%Weights
	Out<-NeuronSum # The output for each case in each row
	Error<-Target-Out
	for (i in 1:(NumInputs+1)) Delta[i]<-sum(Lr*Error*NeuronIn[,i])/100
	Weights<-Weights+Delta
}

