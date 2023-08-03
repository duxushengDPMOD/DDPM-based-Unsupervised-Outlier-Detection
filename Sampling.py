import torch
from Forward_Results import x_t

cell_sampling_results = []
def sampleT(model, args, test_x):
    model.eval()
    with torch.no_grad():
        """From x[T] to x[T-1]、x[T-2]|...x[0]"""

        T = torch.tensor([args.num_steps-1]).to(args.device)
        xt, z = x_t(test_x, T, args)
        for i in reversed(range(args.num_steps)):

            t = torch.tensor([i]).to(args.device)

            predicted_noise = model(xt, t)

            alphat = args.alphas[t].to(args.device)
            one_minus_alphat_bar_sqrt = args.one_minus_alphas_bar_sqrt[t].to(args.device)
            sigmat = args.betas[t].to(args.device)

            if i > 0:
                xt = (1 / alphat.sqrt()) * (xt - (sigmat / one_minus_alphat_bar_sqrt) * predicted_noise) + sigmat * z
            else:
                xt = (1 / alphat.sqrt()) * (xt - (sigmat / one_minus_alphat_bar_sqrt) * predicted_noise)


            Temp_For_Save_SamplingResults=xt
            Temp_For_Save_SamplingResults=Temp_For_Save_SamplingResults.cpu().detach()
            Temp_For_Save_SamplingResults=Temp_For_Save_SamplingResults.numpy()
            cell_sampling_results.append(Temp_For_Save_SamplingResults)

        x0 = xt


    return x0, xt, z,cell_sampling_results