# Implements the algorithm seen in "Linear-Least-Squares Initialization of 
# Multilayer Perceptrons Through Backpropagation of the Desired Response
# Deniz et al (2005), IEEE Transactions on Neural Networks, 16-2.
# 
# Author: Gustavo (lgbpinho153@yahoo.com.br)
###############################################################################

# simple data
I<-rbind(rnorm(100,0,1),rnorm(100,0,1),rnorm(100,0,1)) # inputs
T<-10+2*I[1,]+3*I[2,]+5*I[3,] # targets

### Network parameters
NumInputs<-dim(I)[1] # number of inputs
NumHidden<-2 # number of hidden neurons
NumOutputs<-1 # number of outputs
WeightsHI<-matrix(rnorm(NumHidden*(NumInputs),0,1),NumHidden,NumInputs) # W1: weights between input and hidden layers
BiasHI<-rnorm(NumHidden,0,1) #b1: bias for the first layer
WeightsOH<-matrix(rnorm(NumOutputs*(NumHidden),0,1),NumOutputs,NumHidden) # W2: weights between hidden and output layers
BiasOH<-rnorm(NumOutputs,0,1) #b2: bias for second layer
SumHidden<-rep(0,NumHidden) # the induced field in each hidden neuron
SumOutput<-rep(0,NumOutputs) # the induced field in each output neuron
DeltaH<-rep(0,NumHidden) # local gradient for hidden neurons
DeltaO<-rep(0,NumOutputs) # local gradient for output neurons
DeltaWHI<-matrix(0,NumHidden,NumInputs+1) # weight update matrix for the Hidden-Inputs weights
DeltaWOH<-matrix(0,NumOutputs,NumHidden+1) # weight update matrix for the Output-Hidden weights
Lr<-0.1 # learning rate
Mf<-0.1 # momentum constant
Epochs<-5000
N<-length(T) # number of examples presented to the net
MSEplot<-rep(0,Epochs) # I'll plot the errors later

### preprocessing
Targets<-rep(0,length(T))
Targets<-0.9999*(T-min(T)+0.0001)/(max(T)-min(T)) # standardized outputs
Inputs<-matrix(0,NumInputs,N) # standardized inputs
for (i in 1:dim(Inputs)[1]){
	Inputs[i,]<-(I[i,]-min(I[i,]))/(max(I[i,])-min(I[i,]))
}
HiddenInput<-rbind(1,Inputs)

### Sigmoid Function
phi<-function(v){
	1/(1+exp(-v))
}

phidash<-function(v){
	phi(v)*(1-phi(v))
}

inversephi<-function(v){
	log(v/(1-v))
}

### Weight Initialization

## Step 1
d2<-Targets
d2bar<-inversephi(d2)

## Step 2
d1<-matrix(0,NumHidden,N)
for (i in 1:N){
	d1[,i]<-qr.solve(WeightsOH,(d2bar-BiasOH)[i])	
}
L<-matrix(c(1/(max(d1[1,]-min(d1[1,]))),0,0,1),2)  # set by hand for now
S<-rep(0.4,N)
d1<-L%*%d1+S
d1[d1>=1]<-0.99999;d1[d1<=0]<-0.00001 #gambiarra

## Step 3
d1bar<-inversephi(d1)

## Step 4
W1opt<-matrix(0,NumHidden,NumInputs+1) # candidate for optimum start
for (i in 1:NumHidden){
	b<-0
	for (k in (1:N)){
		b<-b+(phidash(d1bar[i,k])^2)*d1[i,k]*HiddenInput[,k]
	}
	A<-matrix(0,NumInputs+1,NumInputs+1)
	for (k in (1:N)){
		A<-A+(phidash(d1bar[i,k])^2)*HiddenInput[,k]%*%t(HiddenInput[,k])
	}
	wk<-solve(A+0.0000001*diag(NumInputs+1),b)
	W1opt[i,]<-wk	
}

## Step 5
z1<-W1opt%*%HiddenInput
y1<-phi(z1)

## Step 6
W2opt<-matrix(0,NumOutputs,NumHidden+1) # candidate for optimum start
y1<-rbind(1,y1)
b<-0
for (k in (1:N)){
	b<-b+(phidash(d2bar[k])^2)*d2[k]*y1[,k]
}
A<-matrix(0,NumHidden+1,NumHidden+1)
for (k in (1:N)){
	A<-A+(phidash(d2bar[k]))^2*y1[,k]%*%t(y1[,k])
}
wk<-solve(A+0.0000001*diag(NumHidden+1),b)
W2opt<-wk	