# Neural Network with one hidden layer using the backpropagation
# algorithm. Sigmoid activation function. For one output only (for now).
# Author: Gustavo (lgbpinho153@yahoo.com.br)
###############################################################################

### simple data
Inputs<-rbind(rnorm(100,0,0.2),rnorm(100,0,0.2),rnorm(100,0,0.2))
Targets<-10+2*Inputs[1,]+3*Inputs[2,]+5*Inputs[3,]


### Network parameters
NumInputs<-dim(Inputs)[1]
NumHidden<-3
NumOutputs<-1
WeightsHI<-matrix(0,NumHidden,NumInputs+1) # bias and weights between input and hidden layers
WeightsOH<-matrix(0,NumOutputs,NumHidden+1) # bias and weights between hidden and output layers
SumHidden<-rep(0,NumHidden) # the induced field in each neuron
SumHidden<-rep(0,NumOutputs) # the induced field in each output neuron
Lr<-0.05 # learning rate
Mf<-0.05 # momentum factor
Epochs<-10
N<-length(Targets) # number of examples presented to the net

### Sigmoid Function
phi<-function(v){
	1/(1+exp(-v))
}

phidash<-function(v){
	phi(v)*(1-phi(v))
}

### Training
for (s in 1:Epochs){
	for (t in 1:N){
	### Forward Pass
	HiddenInput<-c(1,Inputs[,t])
	SumHidden<-WeightsHI%*%HiddenInput
	HiddenOut<-phi(SumHidden)
	OutputInput<-c(1,HiddenOut) # this is the input for the output layer
	SumOutput<-WeightsOH%*%OutputInput
	Outputs<-phi(SumOutput)
	Error<-Targets[t]-Outputs
	### Backpropagation
	DeltaO<-Error*phidash(SumOutput) # local gradients for outputs neurons
	DeltaWOH<-Lr*DeltaO*OutputInput
	DeltaH<-DeltaO*WeightsOH
	}
}
