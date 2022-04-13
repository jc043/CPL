import torch  
from torch import nn
import torch.nn.functional as F
from core.layers.FCLSTMCell import FCLSTMCell
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell

# From https://github.com/AntixK/PyTorch-VAE
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
class shared_encoder(nn.Module):
    def __init__(self,configs):
        super(shared_encoder, self).__init__()
        self.batch_size=configs.batch_size
        self.img_width=configs.img_width
        in_channels=1
        self.embed_class = nn.Linear(configs.num_classes, configs.img_width * configs.img_width)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        modules = []

        hidden_dims = [32, 64, 128, 256, 512]
        in_channels+=1
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, input,y):
        #[batch, channel=1, height, width]
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_width, self.img_width).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        h1=self.encoder(x)     
        return h1
class head_reconstructor(nn.Module):
    def __init__(self,configs):
        super(head_reconstructor, self).__init__()
        
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        self.decoder_input = nn.Linear(configs.zdim+configs.num_classes, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())
    def forward(self,z):
        result=self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class latent_encoder(nn.Module):
    def __init__(self,configs):
        super(latent_encoder, self).__init__()
        self.configs=configs
        # self.fc1=nn.Sequential(
        # nn.Linear(configs.zdim+configs.num_classes,configs.zdim),
        # )
        # self.fc2=nn.Sequential(
        # nn.Linear(configs.zdim+configs.num_classes,configs.zdim),
        # )
        self.fc_mu = nn.Linear(512*4, configs.zdim)
        self.fc_var = nn.Linear(512*4, configs.zdim)

    def forward(self, x_hiddens,y):
        result = torch.flatten(x_hiddens, start_dim=1)
        mu=self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

class encoder(nn.Module):
    def __init__(self,configs):
        super(encoder, self).__init__()
        self.configs=configs
        self.in_channel = configs.patch_size * configs.patch_size
        width = (configs.img_width // configs.patch_size)//4
        self.c1 = nn.Sequential(
                nn.Conv2d(self.in_channel*2,256,5,2,padding=2),
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
        # self.fc2=nn.Sequential(
        # nn.Linear(configs.zdim+configs.num_classes,configs.zdim),
        # nn.Tanh(),
        # )
        # self.fc3=nn.Sequential(
        # nn.Linear(configs.zdim+configs.num_classes,configs.zdim),
        # nn.Tanh(),
        # )
        self.FCLSTM=FCLSTMCell(256, 256, width, configs.layer_norm)
    def forward(self, input, hidden, cell,y):
        h1 = self.c1(input) 
        h2 = self.c2(h1) 
        h2=h2.flatten(start_dim=1)
        hidden, cell = self.FCLSTM(h2, hidden, cell)
        out=self.fc1(hidden)
        mu,logvar=torch.split(out,self.configs.zdim,dim=1)
        return mu, logvar, hidden, cell

class Prior(nn.Module):
    def __init__(self,configs):
        super(Prior, self).__init__()
        self.configs=configs
        self.in_channel = configs.patch_size * configs.patch_size
        width = (configs.img_width // configs.patch_size)//4
        self.c1 = nn.Sequential(
                nn.Conv2d(self.in_channel*2,256,5,2,padding=2),
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
        # self.fc2=nn.Sequential(
        # nn.Linear(configs.zdim+configs.num_classes,configs.zdim),
        # nn.Tanh(),
        # )
        # self.fc3=nn.Sequential(
        # nn.Linear(configs.zdim+configs.num_classes,configs.zdim),
        # nn.Tanh(),
        # )
        self.FCLSTM=FCLSTMCell(256, 256, width, configs.layer_norm)
    def forward(self, input, hidden, cell,y):
        h1 = self.c1(input) 
        h2 = self.c2(h1) 
        h2=h2.flatten(start_dim=1)
        hidden, cell = self.FCLSTM(h2, hidden, cell)
        out=self.fc1(hidden)
        mu,logvar=torch.split(out,self.configs.zdim,dim=1)

        return mu, logvar, hidden, cell


class Prior_FP(nn.Module):
    def __init__(self,configs):
        self.configs=configs
        super(Prior_FP, self).__init__()
        self.fc1=nn.Linear(configs.num_classes,2*configs.zdim)
        self.fc2=nn.Linear(2*configs.zdim,2*configs.zdim)
    def forward(self,y):
        out=self.fc1(y)
        mu,logvar=torch.split(out,self.configs.zdim,dim=1)

        return mu, logvar
        
class decoder(nn.Module):
    def __init__(self,num_layers,num_hidden,configs):
        super(decoder, self).__init__()
        self.num_layers = num_layers
        self.configs = configs
        self.num_hidden = num_hidden
        self.frame_channel = configs.patch_size * configs.patch_size
        self.padding = configs.filter_size // 2
        self.fc1 = nn.Sequential(
                nn.Linear(configs.zdim,256),
                nn.Tanh(),
                )
        self.fc2=nn.Sequential(
                nn.Linear(256,512),
                nn.Tanh(),
                )
        self.conv=nn.Conv2d(num_hidden,self.frame_channel,
        kernel_size=1, stride=1, padding=0, bias=False)
        cell_list=[]
        width = configs.img_width // configs.patch_size
        self.conv_h=nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=configs.filter_size, stride=configs.stride, padding=self.padding, bias=True),
                nn.LayerNorm([num_hidden, width, width])
            )
        self.conv_z=nn.Sequential(
                nn.Conv2d(configs.zdim+configs.num_classes, num_hidden, kernel_size=configs.filter_size, stride=configs.stride, padding=self.padding, bias=True),
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