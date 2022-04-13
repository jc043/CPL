import torch  
from torch import nn
from core.layers.FCLSTMCell import FCLSTMCell
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell


class encoder(nn.Module):
    def __init__(self,configs):
        super(encoder, self).__init__()
        self.configs=configs
        self.in_channel = configs.patch_size * configs.patch_size
        width = (configs.img_width // configs.patch_size)//4
        self.c1 = nn.Sequential(
                nn.Conv2d(self.in_channel,256,5,2,padding=2),
                nn.ReLU(True),
                )
        self.c2 = nn.Sequential(
                nn.Conv2d(256,256,4,2,padding=1),
                nn.ReLU(True),
                )
        self.fc1=nn.Sequential(
        nn.Linear(256,2*configs.zdim),
        nn.Tanh(),
        )
        self.FCLSTM=FCLSTMCell(256, 256, width, configs.layer_norm)
    def forward(self, input, hidden, cell):
        h1 = self.c1(input) 
        h2 = self.c2(h1) 
        h2 = h2.flatten(start_dim=1)
        hidden, cell = self.FCLSTM(h2, hidden, cell)
        out = self.fc1(hidden)
        mu, logvar = torch.split(out,self.configs.zdim,dim=1)
        return mu, logvar, hidden, cell

class Prior(nn.Module):
    def __init__(self,configs):
        super(Prior, self).__init__()
        self.configs=configs
        self.in_channel = configs.patch_size * configs.patch_size
        width = (configs.img_width // configs.patch_size)//4
        self.c1 = nn.Sequential(
                nn.Conv2d(self.in_channel,256,5,2,padding=2),
                nn.ReLU(True),
                )
        self.c2 = nn.Sequential(
                nn.Conv2d(256,256,4,2,padding=1),
                nn.ReLU(True),
                )
        self.fc1 = nn.Sequential(
        nn.Linear(256,2*configs.zdim),
        nn.Tanh(),
        )
        self.FCLSTM=FCLSTMCell(256, 256, width, configs.layer_norm)
    def forward(self, input, hidden, cell):
        h1 = self.c1(input) 
        h2 = self.c2(h1) 
        h2=h2.flatten(start_dim=1)
        hidden, cell = self.FCLSTM(h2, hidden, cell)
        out=self.fc1(hidden)
        mu,logvar=torch.split(out,self.configs.zdim,dim=1)

        return mu, logvar, hidden, cell
        

class decoder(nn.Module):
    def __init__(self,num_layers,num_hidden,configs):
        super(decoder, self).__init__()
        self.num_layers = num_layers
        self.configs = configs
        self.num_hidden = num_hidden
        self.frame_channel = configs.patch_size * configs.patch_size
        self.padding = configs.filter_size // 2
        
        self.conv=nn.Conv2d(num_hidden,self.frame_channel,
        kernel_size=1, stride=1, padding=0, bias=False)
        cell_list=[]
        width = configs.img_width // configs.patch_size
        
        self.conv_z=nn.Sequential(
                nn.Conv2d(configs.zdim, num_hidden, kernel_size=configs.filter_size, stride=configs.stride, padding=self.padding, bias=True),
                nn.LayerNorm([num_hidden, width, width])
            )
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden
            in_channel+=num_hidden
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden, width, configs.filter_size,
                                       configs.stride, configs.layer_norm)

            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self,images,noise,hidden,cell,memory):
        #image&noise: NCHW
        batch = images.shape[0]
        height = images.shape[2]
        width = images.shape[3]

        noise=noise.repeat(height,width,1,1)
        z_ = noise.permute(2,3,0,1).contiguous()
        z_=self.conv_z(z_)
        
        inputs=torch.cat((images, z_), 1)
        hidden[0], cell[0], memory = self.cell_list[0](inputs, hidden[0], cell[0], memory)
        for i in range(1, self.num_layers):
            inputs = torch.cat((hidden[i-1], z_), 1)
            hidden[i], cell[i], memory  = self.cell_list[i](inputs, hidden[i], cell[i], memory)
        
        x_gen=self.conv(hidden[self.num_layers-1])
        return hidden,cell,x_gen, memory