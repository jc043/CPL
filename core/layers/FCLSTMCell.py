import torch
import torch.nn as nn

class FCLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, layer_norm=True):
        super(FCLSTMCell, self).__init__()
        #num_hidden in FCLSTM is different from STLSTM
        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        self.ct_weight=torch.nn.Parameter(torch.FloatTensor(self.num_hidden*2))
        self.oc_weight=torch.nn.Parameter(torch.FloatTensor(self.num_hidden))
        if layer_norm:
            self.fc_h=nn.Sequential(
            nn.Linear(in_channel,self.num_hidden * 4),
            nn.LayerNorm([self.num_hidden * 4])
            )
            # input batch*(256*4*4)
            self.fc_x=nn.Sequential(
            nn.Linear(in_channel*width*width,self.num_hidden * 4),
            nn.LayerNorm([self.num_hidden * 4])
            )
        else:
            self.fc_h=nn.Sequential(
            nn.Linear(in_channel,self.num_hidden * 4),
            )
            self.fc_x=nn.Sequential(
            nn.Linear(in_channel*width*width,self.num_hidden * 4),
            )
    def forward(self, x_t, h_t, c_t):
        #print("x_t"+str(x_t.shape))
        #print(h_t.shape)
        h_concat = self.fc_h(h_t)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        
        ct_activation=torch.mul(c_t.repeat(1,2),self.ct_weight)
        i_c, f_c = torch.split(ct_activation,self.num_hidden, dim=1)
        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x_t != None:
            x_concat=self.fc_x(x_t)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ =i_+ i_x
            f_ =f_+ f_x
            g_ =g_+ g_x
            o_ =o_+ o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)
        
        o_c=torch.mul(c_new,self.oc_weight)
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new
        
