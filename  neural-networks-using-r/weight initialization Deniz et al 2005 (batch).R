# Implements the algorithm seen in "Linear-Least-Squares Initialization of 
# Multilayer Perceptrons Through Backpropagation of the Desired Response
# Deniz et al (2005), IEEE Transactions on Neural Networks, 16-2.
# 
# Author: Gustavo (lgbpinho153@yahoo.com.br)
###############################################################################

# simple data
I<-rbind(rnorm(100,0,1),rnorm(100,0,1),rnorm(100,0,1)) # inputs
T<-10+2*I[1,]+3*I[2,]+5*I[3,] # targets

# parameters
NumInput<-dim(I)[1] # the number of inputs is the number of rown in I
NumOutput<-1 # keep it simple for now
NumHidden<-2 # number of hidden neurons
WeightsOH<-matrix(rnorm(NumOutput*(NumHidden+1),0,1),NumOutput,NumHidden+1) # weights and bias for the output-hidden layer
WeightsHI<-matrix(rnorm(NumHidden*(NumInput+1),0,1),NumHidden,NumInput+1)  # weights and bias for the hidden-input layer
N<-length(T) # number of samples
Lr<-0.2 # learning rate
Mr<-0.15 # momentum rate
NumEpochs<-5000
MSE<-rep(0,NumEpochs)

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


### preprocessing
Targets<-rep(0,length(T))
Targets<-(T-min(T))/(max(T)-min(T)) # standardized outputs
Inputs<-matrix(0,NumInput,N) # standardized inputs
for (i in 1:dim(Inputs)[1]){
	Inputs[i,]<-(I[i,]-min(I[i,]))/(max(I[i,])-min(I[i,]))
}

##########################
# Weights initialization #
##########################

# Step 0 #
xs<-Inputs
d2<-Targets
W1<-WeightsHI[,-1]
b1<-WeightsHI[,1]
W2<-WeightsOH[,-1]
b2<-WeightsOH[,1]
z1<-W1%*%xs+b1
y1<-phi(z1)
z2<-W2%*%y1+b2
y2<-phi(z2)
Jopt<-sum(t(d2-y2)%*%(d2-y2))/(2*N)
W1opt<-W1
b1opt<-b1
W2opt<-W2
b2opt<-b2

# Step 1 #
d2[d2==1]<-1-1e-3;d2[d2==0]<-1e-3
d2bar<-inversephi(d2)

for (t in 1:5){ # this is step 9
	
# Step 2 #
	d1<-matrix(0,NumHidden,N)
	W2<-t(W2) # small issue with R matrices and vectors
	for (i in 1:N){
		d1[,i]<-qr.solve(W2,d2bar[i]-b2)	
	}
	d1[1,]<-(d1[1,]-min(d1[1,]))/(max(d1[1,])-min(d1[1,]))
	d1[d1==1]<-1-1e-3;d1[d1==0]<-1e-3
	
# Step 3 #
	d1bar<-inversephi(d1)
	
# Step 4 # It is like solving a Ax=b problem
	for (k in 1:NumHidden){ # for each hidden neuron
		b<-matrix(0,NumInput+1,1)
		A<-matrix(0,NumInput+1,NumInput+1)
		for (s in 1:N){
			us<-c(1,xs[,s])
			b<-b+(phidash(d1bar[k,s])^2)*d1[k,s]*us
			A<-A+(phidash(d1bar[k,s])^2)*us%*%t(us)
		}
		wk<-solve(A,b)
		b1[k]<-wk[1,]
		W1[k,]<-wk[2:(NumInput+1),]
	}
	
# Step 5 #
	z1<-W1%*%xs+b1
	y1<-phi(z1)
	
# Step 6 # Only one output neuron
	b<-matrix(0,NumHidden+1,1)
	A<-matrix(0,NumHidden+1,NumHidden+1)
	for (s in 1:N){
		us<-c(1,y1[,s])
		b<-b+(phidash(d2bar[s])^2)*d2[s]*us
		A<-A+(phidash(d2bar[s])^2)*us%*%t(us)
	}
	wk<-solve(A+diag(0.001,NumHidden+1),b) # A may be singular
	b2<-wk[1,]
	W2<-wk[2:(NumHidden+1),]
	
# Step 7 #
	z2<-W2%*%y1+b2
	y2<-phi(z2)
	
# Step 8 #
	J<-sum(t(d2-y2)%*%(d2-y2))/(2*N)
	if(J<Jopt){
		Jopt<-J
		W1opt<-W1
		b1opt<-b1
		W2opt<-W2
		b2opt<-b2	
	}
}

WeightsHI<-cbind(b1,W1)
WeightsOH<-t(c(b2,W2)) # W2 is a vector
################################
# End of Weight Initialization #
################################

############
# Training #
############
DeltaWOH<-matrix(0,NumOutput,NumHidden+1) # weights updates
DeltaWHI<-matrix(0,NumHidden,NumInput+1)  # weights updates
GradNorm<-rep(0,NumEpochs)
for (epoch in 1:NumEpochs){
	PreviousDeltaWOH<-DeltaWOH # storing previous values for momentum
	PreviousDeltaWHI<-DeltaWHI # storing previous values for momentum
	DeltaWOH<-matrix(0,NumOutput,NumHidden+1) # weights updates
	DeltaWHI<-matrix(0,NumHidden,NumInput+1)  # weights updates
	
	### Forward pass
	HiddenInputs<-rbind(1,Inputs)
	SumHidden<-WeightsHI%*%HiddenInputs
	HiddenOutputs<-phi(SumHidden)
	OutputInputs<-rbind(1,HiddenOutputs)
	SumOutput<-WeightsOH%*%OutputInputs
	Outputs<-phi(SumOutput)
	MSE[epoch]<-as.numeric(t(Targets-t(Outputs))%*%(Targets-t(Outputs)))/(2*N)
	
	### Backpropagation
	DeltaO<-(Targets-t(Outputs))*t(Outputs)*(1-t(Outputs)) # the local gradients for each sample
	DeltaH<-matrix(0,NumHidden,N)
	for (s in 1:N){ # hidden layer local gradients for each example 
		for (i in 1:NumHidden){
			for (j in 1:NumOutput){ # gotta fix for more neurons later
				DeltaH[i,s]<-DeltaH[i,s]+DeltaO[s]*WeightsOH[j,i+1] # beware of the bias index!	
			}
		}
	}
	for(s in 1:N){ # output-hidden weights updates
		for (j in 1:NumOutput){
			for (i in 1:(NumHidden+1)){
				DeltaWOH[j,i]<-DeltaWOH[j,i]+(Lr*DeltaO[s]*OutputInputs[i,s])/N
			}	
		}
	}
	for(s in 1:N){ # hidden-input weights updates
		for (j in 1:NumHidden){
			for (i in 1:(NumInput+1)){
				DeltaWHI[j,i]<-DeltaWHI[j,i]+(Lr*DeltaH[j,s]*HiddenInputs[i,s])/N
			}	
		}
	}
	Grad<-c(DeltaWOH,DeltaWHI)/Lr
	GradNorm[epoch]<-sqrt(t(Grad)%*%Grad)
	WeightsOH<-WeightsOH+DeltaWOH+Mr*PreviousDeltaWOH
	WeightsHI<-WeightsHI+DeltaWHI+Mr*PreviousDeltaWHI
	if(epoch%%100==0){
		cat("Epoch: ");cat(epoch);cat(" | MSE: ");cat(MSE[epoch]);cat(" | Gradient: ");cat(GradNorm[epoch]);cat("\n");
	}
}
par(mfrow=c(2,2))
plot(MSE,main="MSE",xlab="Epoch",ylab="MSE",type='l',col='red')
plot(GradNorm,main="Gradient",xlab="Epoch",ylab="Gradient",type='l',col='red')
plot(Targets,t(Outputs),main="Fit",xlab="Targets",ylab="Outputs",pch=16)
abline(lm(Targets~t(Outputs)),col='red')
hist(Targets-t(Outputs),main="Error",xlab="Error",ylab="Frequency",col='lightblue')