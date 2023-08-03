import torch
def x_t(x_0, t, args):


    x_0 = x_0.to(args.device)
    noise = torch.randn_like(x_0).to(args.device)
    alphas_t = args.alphas_bar_sqrt[t].to(args.device)
    alphas_1_m_t = args.one_minus_alphas_bar_sqrt[t].to(args.device)

    return (alphas_t * x_0 + alphas_1_m_t * noise), noise