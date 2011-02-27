# Neural Network with one hidden layer using the backpropagation
# algorithm. Sigmoid activation function. For one output only (for now).
# Batch mode.
# Author: Gustavo (lgbpinho153@yahoo.com.br)
# TODO: Validation and test sets
###############################################################################

start<-Sys.time() # counter for execution time
### simple data
I<-rbind(rnorm(100,0,1),rnorm(100,0,1),rnorm(100,0,1)) # inputs
T<-10+2*I[1,]+3*I[2,]+5*I[3,] # targets

### Network parameters
NumInputs<-dim(I)[1] # number of inputs
NumHidden<-2 # number of hidden neurons
NumOutputs<-1 # number of outputs
WeightsHI<-matrix(rnorm(NumHidden*(NumInputs+1),0,1),NumHidden,NumInputs+1) # bias and weights between input and hidden layers
WeightsOH<-matrix(rnorm(NumOutputs*(NumHidden+1),0,1),NumOutputs,NumHidden+1) # bias and weights between hidden and output layers
SumHidden<-rep(0,NumHidden) # the induced field in each hidden neuron
SumOutput<-rep(0,NumOutputs) # the induced field in each output neuron
DeltaH<-rep(0,NumHidden) # local gradient for hidden neurons
DeltaO<-rep(0,NumOutputs) # local gradient for output neurons
DeltaWHI<-matrix(0,NumHidden,NumInputs+1) # weight update matrix for the Hidden-Inputs weights
DeltaWOH<-matrix(0,NumOutputs,NumHidden+1) # weight update matrix for the Output-Hidden weights
Lr<-0.15 # learning rate
Mf<-0.15# momentum constant
Epochs<-5000
N<-length(T) # number of examples presented to the net
MSEplot<-0 # I'll plot the errors later

### preprocessing
Targets<-rep(0,length(T))
Targets<-(T-min(T))/(max(T)-min(T)) # standardized outputs
Inputs<-matrix(0,NumInputs,N) # standardized inputs
for (i in 1:dim(Inputs)[1]){
	Inputs[i,]<-(I[i,]-min(I[i,]))/(max(I[i,])-min(I[i,]))
}


### Sigmoid Function
phi<-function(v){
	1/(1+exp(-v))
}

phidash<-function(v){
	phi(v)*(1-phi(v))
}

### Training
for (s in 1:Epochs){ # for each epoch
	DeltaWHIPrevious<-DeltaWHI # storing the previous updates
	DeltaWOHPrevious<-DeltaWOH # storing the previous updates
	DeltaH<-rep(0,NumHidden) # reset
	DeltaO<-rep(0,NumOutputs) # reset
	DeltaWHI<-matrix(0,NumHidden,NumInputs+1) # reset
	DeltaWOH<-matrix(0,NumOutputs,NumHidden+1) # reset
	MSE<-0 #mean squared error
	GradientNorm<-0 # the euclidian norm for the gradient vector
	### Forward Pass
	HiddenInput<-rbind(1,Inputs) # inputs for the hidden layer
	SumHidden<-WeightsHI%*%HiddenInput
	HiddenOut<-phi(SumHidden)
	OutputInput<-rbind(1,HiddenOut) # this is the input for the output layer
	SumOutput<-WeightsOH%*%OutputInput
	Outputs<-phi(SumOutput)
	Error<-Targets-Outputs #corrigir isso para mais neuronios
	MSE<-0.5*Error%*%t(Error)/N
		### Backpropagation
	for (k in 1:N){
		for (i in 1:NumOutputs){
			DeltaO[i]<-Error[i]*phidash(SumOutput[i])
		}
		for (i in 1:NumOutputs){  # for each output neuron
			for (j in 1:NumHidden+1){ # for each weight
				DeltaWOH[i,j]<-DeltaWOH[i,j]+(Lr*DeltaO[i]*OutputInput[j])/N
				GradientNorm<-GradientNorm+((DeltaO[i]*OutputInput[j])^2)/N
			}
		}
		for (i in 1:(NumHidden)){ # local gradients for hidden layer
			for (j in 1:NumOutputs){
				DeltaH[i]<-DeltaH[i]+DeltaO[j]*WeightsOH[j,i+1] # beware of teh bias
			}
			DeltaH[i]<-phidash(SumHidden[i])*DeltaH[i]
		}
		for (i in 1:NumHidden){
			for (j in 1:NumInputs+1){
				DeltaWHI[i,j]<-DeltaWHI[i,j]+(Lr*DeltaH[i]*HiddenInput[j])/N
				GradientNorm<-GradientNorm+((DeltaH[i]*HiddenInput[j])^2)/N
			}
		}		
	}	
		WeightsOH<-WeightsOH+DeltaWOH+Mf*DeltaWOHPrevious
		WeightsHI<-WeightsHI+DeltaWHI+Mf*DeltaWHIPrevious
		GradientNorm<-sqrt(GradientNorm)
		MSEplot[s]<-MSE
		if(s%%100==0) {
			cat("MSE at epoch ");cat(s);cat(" is: ");cat(MSE);
			cat(" | Gradient is: ");cat(GradientNorm);cat("\n");
		}
		if (GradientNorm<1e-10) {cat(Error%*%t(Error));cat("\n");break}
	}
plot(MSEplot)
#print(WeightsOH)
#print(WeightsHI)
stop<-Sys.time()
time<-start-stop
cat("Tempo gasto: ");cat(stop-start);cat("\n")

### Simulation

### new data
I<-rbind(rnorm(100,0,1),rnorm(100,0,1),rnorm(100,0,1)) # inputs
T<-10+2*I[1,]+3*I[2,]+5*I[3,] # targets

### preprocessing
Targets<-rep(0,length(T))
Targets<-(T-min(T))/(max(T)-min(T)) # standardized outputs
Inputs<-matrix(0,NumInputs,N) # standardized inputs
for (i in 1:dim(Inputs)[1]){
	Inputs[i,]<-(I[i,]-min(I[i,]))/(max(I[i,])-min(I[i,]))
}


OutputsSim<-rep(0,N)
for (t in 1:N){ # for each example 
	### Forward Pass
	HiddenInput<-c(1,Inputs[,t]) # inputs for the hidden layer
	SumHidden<-rep(0,NumHidden)
	for (i in 1:NumHidden){ # for each neuron 
		for (j in 1:NumInputs+1){ # for each input
			SumHidden[i]<-SumHidden[i]+WeightsHI[i,j]*HiddenInput[j]
		}
	}
	HiddenOut<-phi(SumHidden)
	OutputInput<-c(1,HiddenOut) # this is the input for the output layer
	SumOutput<-rep(0,NumOutputs)
	for (i in 1:NumOutputs){ # for each neuron 
		for (j in 1:NumHidden+1){ # for each input
			SumOutput[i]<-SumOutput[i]+WeightsOH[i,j]*OutputInput[j]
		}
	}
	OutputsSim[t]<-phi(SumOutput)
}
ErrorsSim<-Targets-OutputsSim
#X11()
#plot(ErrorsSim)
cat(0.5*sum(ErrorsSim^2)/N);cat("\n")
