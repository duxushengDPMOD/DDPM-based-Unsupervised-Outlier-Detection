import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from DiffusionModel import DiffusionModel
import torch.nn as nn
import torch
from Forward_Results import x_t
from Sampling import sampleT
from Visulization import show_result_test
from Visulization import show_allObject_loss
from sklearn.decomposition import PCA
from Visulization import show_PCA_Results
from GenerateNewSample import Generate_New_Sample

def train(args, train_x,num_epoch,lr):
    import time


    loss = 0.0
    running = 0.0
    s = 1
    model = DiffusionModel(args, train_x).to(args.device)
    batch = train_x.shape[0]
    # batch = 512
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    train_x = torch.tensor(train_x).float()

    starttime = time.time()
    loss_list = []
    for epoch in range(num_epoch):
        for i in range(train_x.shape[0] // batch):
            input_batch = train_x[i * batch: (i + 1) * batch]


            t = torch.randint(0, args.num_steps, size=(input_batch.shape[0],)).to(args.device)
            t = t.unsqueeze(-1)


            x, noise = x_t(input_batch, t, args)

            output = model(x, t.squeeze(-1))



            noise_loss = mse(noise, output)

            noise_loss_value = noise_loss.item()

            loss_list.append(noise_loss_value)

            optimizer.zero_grad()
            noise_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            running += noise_loss.data.cpu().numpy()


        running = 0.0
        print('epoch number:', epoch)

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(args.data).float()

        x_0, xt, z,cell_sampling_results = sampleT(model, args, test_x)
        x_0 = x_0.cpu().detach()

    return model, x_0,train_x,loss_list,cell_sampling_results
