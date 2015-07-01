# Standard Backpropagation FeedForward Neural Network

###############################################################################

# Samples 
##########

# I<-cbind(runif(100),runif(100),runif(100))
# T<-matrix(10 
#           +2*I[,1]
#           +3*I[,2]
#           +5*I[,3],100,1)

#Exclusive XOR
I <- cbind(c(0,0,1,1), c(0,1,0,1))   # Input
T <- cbind(c(0,1,1,0))               # Output

#
# Parameters

N <- nrow(I) # number of samples
NumInputs <- ncol(I) # Number of variables. 
NumHidden <- 2 # number of hidden layers
NumOutputs <- ncol(T)

# Weights from Input to Hidden
W1 <- matrix(runif(NumHidden*NumInputs),NumInputs,NumHidden)
b1 <- matrix(runif(NumHidden),1,NumHidden)
W1bar<- rbind(W1,b1)

# Weights from hidden to Output Layer
W2 <- matrix(runif(NumOutputs*NumHidden),NumHidden,NumOutputs)
b2 <- matrix(runif(NumOutputs),1,NumOutputs)
W2bar <- rbind(W2,b2)


Lr<-0.05  # Learning Rate
Mf<-0.05  # ??
NumEpochs<-1000


# Pre-process

Inputs <- matrix(0,N,NumInputs)
Targets <- matrix(0,N,NumOutputs)

for (i in 1:NumInputs){
	Inputs[,i] <- 0.99 * (I[,i]-min(I[,i]))/(max(I[,i])-min(I[,i])) + 0.001	
}
for (i in 1:NumOutputs){
	Targets[,i] <- 0.99 * (T[,i]-min(T[,i]))/(max(T[,i])-min(T[,i])) + 0.001	
}

# Sigmoid
phi <- function(v) 1/(1+exp(-v))
phidash <- function(v) phi(v)*(1-phi(v))

# Training
MSE <- rep(0,NumEpochs)
Grad <- rep(0,NumEpochs)
MinGrad <- 1
PreviousDW1bar <- 0
PreviousDW2bar <- 0

for (epoch in 1:NumEpochs){
	for(s in 1:N){ # adjusts are made after each sample
		# forward pass
		HiddenInputs <- c(Inputs[s,],1)
		HiddenInducedFields <- HiddenInputs%*%W1bar
		HiddenOutputs <- phi(HiddenInducedFields)
		OutputInputs <- c(HiddenOutputs,1)
		OutputInducedFields <- OutputInputs%*%W2bar
		Outputs <- phi(OutputInducedFields)
		
		#backpropagation
		Delta2<-(Targets[s,]-Outputs)*Outputs*(1-Outputs) # local gradient for the output neurons
		D1<-diag(c(HiddenOutputs*(1-HiddenOutputs))) # activation derivatives
		Delta1<-D1%*%W2%*%Delta2 # local gradient for the hidden neurons		

		DeltaW2bar<-t(Lr*Delta2%*%OutputInputs)
		DeltaW1bar<-t(Lr*Delta1%*%HiddenInputs)
		
		W2bar<-W2bar+DeltaW2bar+Mf*PreviousDW2bar
		W1bar<-W1bar+DeltaW1bar+Mf*PreviousDW1bar
		
		PreviousDW2bar<-DeltaW2bar
		PreviousDW1bar<-DeltaW1bar
		
		W2<-W2bar[1:NumHidden,];b2<-W2bar[NumHidden+1,]
		W1<-W1bar[1:NumInputs,];b1<-W1bar[NumInputs+1,]
	}
	
	# simulating outputs
	HiddenInputsSim<-cbind(Inputs,1)
	HiddenInducedFieldsSim<-HiddenInputsSim%*%W1bar
	HiddenOutputsSim<-phi(HiddenInducedFieldsSim)
	OutputInputsSim<-cbind(HiddenOutputsSim,1)
	OutputInducedFieldsSim<-OutputInputsSim%*%W2bar
	OutputsSim<-phi(OutputInducedFieldsSim)
	
	MSE[epoch]<-t(Targets-OutputsSim)%*%(Targets-OutputsSim)/(2*N)  ### ajeitar isso depois
	Grad[epoch]<-sqrt(sum(c(DeltaW1bar,DeltaW2bar)^2)/(Lr^2))
	
	if(Grad[epoch]<MinGrad){ # Get optimum epoch
		OptW2bar<-W2bar
		OptW1bar<-W1bar
		OptOutputs<-OutputsSim
		MinGrad<-Grad[epoch]
		OptEpoch<-epoch
	}
		
}
par(mfrow=c(2,2))
plot(MSE,type='l',main='Mean Squared Error',col='red',ylab='MSE',xlab="Epoch")
plot(Grad,type='l',main="Gradient",col='red',ylab="Gradient",xlab="Epoch")
plot(Targets,OptOutputs,main='Targets vs. Outputs at \n Best Gradient',ylab="Outputs",xlab="Targets")
abline(lm(Targets~OptOutputs),lty=2,col='red')
hist(Targets-OutputsSim,col='palegreen3',main="Errors",xlab="Error")