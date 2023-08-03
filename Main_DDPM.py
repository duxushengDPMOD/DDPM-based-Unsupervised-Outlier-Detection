import argparse

import matplotlib.pyplot as plt
import torch
from beta_schedule import get_named_beta_schedule
import numpy as np
from Train_DDPM_Model import train
from Measure_Method import Measure
from Visulization import show_PCA_Results
from Visulization import show_result_test,show_allObject_loss
from GenerateNewSample import Generate_New_Sample
from Autoencoder import autoencoder
import time


if __name__ == '__main__':

    start_time = time.time()


    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.units = 500
    args.num_steps = 100
    args.device = "cuda"
    args.betas = get_named_beta_schedule("linear", args.num_steps).to(args.device)
    args.alphas = 1 - args.betas
    args.alphas = args.alphas.to(args.device)
    args.alphas_bar = torch.cumprod(args.alphas, 0)
    args.alphas_bar_sqrt = torch.sqrt(args.alphas_bar).to(args.device)
    args.one_minus_alphas_bar_sqrt = torch.sqrt(1 - args.alphas_bar).to(args.device)


    dataname = 'Normalization_breastw'
    datalabel='Label_breastw'
    args.dataname = dataname
    data = np.loadtxt(f'E:/DiffusionModel_Learning/Data&Label/Normalization_data/{dataname}.txt')
    label = np.loadtxt(f'E:/DiffusionModel_Learning/Data&Label/Label_Normalization_data/{datalabel}.txt')
    args.data = data
    num_epoch=100
    lr=1e-3
    outlier_number=239;


    model, output,train_x,loss_list,cell_sampling_results = train(args, data,num_epoch,lr)
    output = output.cpu().detach()



    data_tensor=torch.tensor(data).float()
    noise_ForNewSample = torch.randn_like(data_tensor).to(args.device)

    x_0_NewSample, xt_NewSample, z_NewSample, NewSample = Generate_New_Sample(model, args, noise_ForNewSample)
    x_0_NewSample = x_0_NewSample.cpu().detach()




    num_epochs=5
    learning_rate=1e-3
    hiddenlayerNeurons=300

    model_AE,output_AE,loss_list=autoencoder(x_0_NewSample,num_epochs,learning_rate,hiddenlayerNeurons)

    args.data=torch.tensor(args.data).float().to(args.device)

    Reconstructed_Original_data=model_AE(args.data)
    Reconstructed_Original_data=Reconstructed_Original_data.cpu().detach()
    Reconstructed_Original_data=Reconstructed_Original_data.numpy()
    Measure(Reconstructed_Original_data, data, label, outlier_number)


    np.savetxt(f'./Reconstructed Data/{dataname}_Reconstructed_by_AE.txt', Reconstructed_Original_data)


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))


    Measure(output,train_x,label,outlier_number)


    end_time = time.time()

    execution_time = end_time - start_time



    np.savetxt(f'{dataname}_NewSampling.txt', x_0_NewSample)



