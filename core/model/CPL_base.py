import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.model.base_networks import encoder
from core.model.base_networks import decoder
from core.model.base_networks import Prior
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell as STLSTM
class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs=configs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.encoder=encoder(configs)
        self.decoder=decoder(num_layers,num_hidden,configs)
        self.prior=Prior(configs)
        self.MSE_criterion = nn.MSELoss()
        
    def reparameterize(self,mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.normal(0,1,size=std.shape,device=self.configs.device)
        return eps * std + mu
    def kl_criterion(self,mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum() / self.configs.batch_size

    def forward(self, frames_tensor,mask_true,category,is_train=True):
        images = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        lenth=images.shape[1]
        batch = images.shape[0]
        height = images.shape[3]
        width = images.shape[4]
        gen_images = []
        cell = []
        hidden = []
        loss_kl = 0
        loss_pd=0
        c_test = []
        h_test = []
        y_gt=torch.eye(self.configs.num_classes,device=self.configs.device)[category]
        if is_train:
            for i in range(self.num_layers):
                zeros = torch.zeros(batch,self.num_hidden,height,width).to(self.configs.device)
                cell.append(zeros)
                hidden.append(zeros)
            memory= torch.zeros(batch,self.num_hidden,height,width).to(self.configs.device)
            zeros = torch.zeros(batch,256).to(self.configs.device)
            h_encode = zeros
            c_encode = zeros
            h_prior = zeros
            c_prior = zeros
        else:  
            for i in range(self.num_layers):  
                zeros = torch.zeros(batch,self.num_hidden,height,width).to(self.configs.device)
                c_test.append(zeros)
                h_test.append(zeros)
            memory= torch.zeros(batch,self.num_hidden,height,width).to(self.configs.device)
            zeros_ = torch.zeros(batch,256).to(self.configs.device)
            h_prior_test = zeros_
            c_prior_test = zeros_
       

        
        if is_train:
            for time_step in range(self.configs.total_length-1):
                if time_step < self.configs.input_length:
                    input_train = images[:,time_step]
                else:
                    input_train = mask_true[:,time_step - self.configs.input_length] * images[:,time_step] \
                            + (1-mask_true[:,time_step- self.configs.input_length]) * x_gen
               
                mu, logvar, h_encode, c_encode = self.encoder(images[:,time_step+1], h_encode, c_encode)

                mu_prior, logvar_prior, h_prior, c_prior = self.prior(
                    input_train, h_prior, c_prior)
                loss_kl += self.kl_criterion(mu, logvar, mu_prior, logvar_prior)
                z_train = self.reparameterize(mu, logvar)
                z_decode = z_train
                hidden, cell, x_gen, memory = self.decoder(
                    input_train, z_decode, hidden, cell,memory)
                loss_pd+=self.MSE_criterion(images[:,time_step+1],x_gen)
                gen_images.append(x_gen)
                
            gen_images = torch.stack(gen_images, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        else:
            y=y_gt.repeat(self.configs.num_samples,1)
            input_length=min(lenth,self.configs.input_length)
            for time_step in range(self.configs.total_length-1):
                if time_step < input_length:
                    input_test = images[:,time_step]
                else:
                    input_test = x_gen_test

                mu, logvar, h_prior_test, c_prior_test = self.prior(
                    input_test, h_prior_test, c_prior_test)
                z_prior = self.reparameterize(mu, logvar)
                h_test, c_test, x_gen_test, memory = self.decoder(
                    input_test, z_prior, h_test, c_test,memory)
                gen_images.append(x_gen_test)
            gen_images = torch.stack(gen_images, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        if loss_kl>10 :
            loss = self.configs.kl_beta * loss_kl + loss_pd
        else:
            loss=loss_pd

        return gen_images, loss, loss_pd,loss_kl
