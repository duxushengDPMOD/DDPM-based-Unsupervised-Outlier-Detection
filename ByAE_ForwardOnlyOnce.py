import torch
from Forward_Results import x_t
import torch.nn as nn

def AE_OnceForward(model,test_x):
    model.eval()
    with torch.no_grad():
        train_data = torch.tensor(train_data).float()
        train_data = train_data.cuda()

        batch_size = train_data.shape[0]


        class autoEncoder(nn.Module):
            def __init__(self):
                super(autoEncoder, self).__init__()
                self.features = train_data.shape[1]
                self.num_units = hiddenlayerNeurons
                self.encoder = nn.Sequential(
                    nn.Linear(self.features, self.num_units),
                    nn.ReLU(True),
                    nn.Linear(self.num_units, self.num_units),
                    nn.ReLU(True),
                    nn.Linear(self.num_units, self.num_units))
                self.decoder = nn.Sequential(
                    nn.Linear(self.num_units, self.num_units),
                    nn.ReLU(True),
                    nn.Linear(self.num_units, self.num_units),
                    nn.ReLU(True),
                    nn.Linear(self.num_units, self.features))

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        model_AE = autoEncoder().cuda()  # autoEncoder model
        loss_func = nn.MSELoss()  # loss function
        optimizer = torch.optim.Adam(model_AE.parameters(), lr=learning_rate, weight_decay=1e-5)
        loss_list = []
        for epoch in range(num_epochs):

            output = model_AE(train_data)
            loss = loss_func(output, train_data)

            loss_value = loss.item()

            loss_list.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))


        output = output.cpu().detach()
        output = output.numpy()
        train_data = train_data.cpu().detach()
        train_data = train_data.numpy()

        return model_AE, output, loss_list