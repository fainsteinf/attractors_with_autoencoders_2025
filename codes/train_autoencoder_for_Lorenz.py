# F. Fainstein (1,2), G. B. Mindlin (1,2), P. Groisman (3). Reconstructing attractors with autoencoders. Chaos, 35(1) (2025).

# 1. Universidad de Buenos Aires, Facultad de Ciencias Exactas y Naturales, Departamento de Física, Ciudad Universitaria, 1428 Buenos Aires, Argentina.
# 2. CONICET - Universidad de Buenos Aires, Instituto de Física Interdisciplinaria y Aplicada (INFINA), Ciudad Universitaria, 1428 Buenos Aires, Argentina.
# 3. IMAS-CONICET and Departamento de Matemática, Facultad de Ciencias Exactas y Naturales, Universidad de Buenos Aires, Ciudad Universitaria, 1428 Buenos Aires, Argentina


import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv
import time


#Define Sine activation subclass
class Sin(torch.nn.Module):  #Esto dice que tiene todas las mismas utilidades que torch.nn.Module
    def forward(self, x):
        return torch.sin(x)


# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__() 

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(40 * 40 * 2, 64),
            Sin(),
            torch.nn.Linear(64, 32),
            Sin(),
            torch.nn.Linear(32, 16),
            Sin(),
            torch.nn.Linear(16, 3)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            Sin(),
            torch.nn.Linear(16, 32),
            Sin(),
            torch.nn.Linear(32, 64),
            Sin(),
            torch.nn.Linear(64, 40 * 40 * 2)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Define loss function
def loss_vel( x, fx, lambda1=1, lambda2=1 ):
    #MSE loss
    mse = torch.sum( (fx - x)**2, dim=-1 )
    #MVE loss (mean velocity loss)
    delta_x = x[1:, :] - x[:-1, :]
    delta_fx =  fx[1:, :] - fx[:-1, :]
    mve = torch.sum(  (delta_fx - delta_x)**2, dim=-1  )
        
    return torch.sum( lambda1 * mse ), torch.sum(lambda2 * mve ) 

#%% get data

#Movie folder
root_dir_movie = '.../'
root_dir_movie += 'lorenz_movie_XYZ.npy'

x_data = np.load(root_dir_movie).astype('float32')

X_2 = x_data.reshape(x_data.shape[0], x_data.shape[1]*x_data.shape[2])

#Substract the mean
X_2 -= np.mean(X_2,axis=0)

print(X_2.shape)

frame_rate = 1
#El tamaño de train size:
train_size = int( 30000 / frame_rate )

#Split train and test
X_train = X_2[:train_size]
X_test = X_2[train_size:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
#%%
start = time.time()

cant_data = X_train.shape[0]
epochs = 400
batch_size = int(cant_data*0.02)
lr = 10**-4

#set loss terms weights
lambda1 = 1
lambda2 = 10
#Set if MVE will be minimized
MVE_on = True

model = AE()

# Using an Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(),
                              lr = lr)

#Follow loss during training
mse, mve = [],[]
mse_test, mve_test = [], []

#Value of loss to follow the best epoch
best_loss_value = 1000

#Evaluate latent space representation
z_lat = []

for epoch in range(0, epochs):
    if epoch%20==0:
        print("Epoca numero: ", epoch)
    epoch_mse, epoch_mve = [], []

    #Set model to train
    model.train()
    for batch in range(cant_data // batch_size):
        image = torch.from_numpy( X_train[ batch_size * batch : batch_size * (batch +1), : ] )

        # Output of Autoencoder
        reconstructed = model(image)

        # Calculating the loss function
        loss_mse, loss_mve = loss_vel(image, reconstructed, lambda1=lambda1, lambda2=lambda2)
        if MVE_on:
            loss_total = loss_mse + loss_mve
        else:
            loss_total = loss_mse + 0 * loss_mve

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        epoch_mse.append(loss_mse.detach().numpy()/batch_size)
        epoch_mve.append(loss_mve.detach().numpy()/batch_size)

    # Storing the losses in a list for plotting
    mse.append(np.mean(epoch_mse))
    mve.append(np.mean(epoch_mve))

    #Set model to evaluation to compute loss for the test set
    model.eval()
    images_test = torch.from_numpy( X_test )
    reconstructed_test = model(images_test)
    loss_mse, loss_mve = loss_vel(images_test, reconstructed_test, lambda1=lambda1, lambda2=lambda2)
    mse_test.append(loss_mse.detach().numpy()/len(X_test))
    mve_test.append(loss_mve.detach().numpy()/len(X_test))

    #Latent space representation
    z = model.encoder(torch.from_numpy(X_test)).detach().numpy() 
    z_lat.append(z)

    #Check if model is the best so far
    test_lossepoch = mse_test[-1]+mve_test[-1]
    if  test_lossepoch < best_loss_value:
        best_model = copy.deepcopy(model)
        best_loss_value = test_lossepoch

end = time.time()
print("The time of execution is :",
      np.round((end-start) / 60, 2), "min")

#Save results
saving_dir = '.../'
ntrain = 0
# np.save(saving_dir+"{}-train_mse.npy".format(ntrain),mse)
# np.save(saving_dir+"{}-test_mse.npy".format(ntrain),mse_test)
# np.save(saving_dir+"{}-train_mve.npy".format(ntrain),mve)
# np.save(saving_dir+"{}-test_mve.npy".format(ntrain),mve_test)

# Latent space representation
# np.save(saving_dir+"{}-z_lat_test.npy".format(ntrain),z_lat)

# Model at best epoch
# torch.save(best_model.state_dict(), saving_dir+"{}-MSE_MVE_model_weights.pth".format(ntrain))
