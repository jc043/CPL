import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from core.model import CPL_base,CPL
import cv2
from core.utils import preprocess
import itertools
import copy
class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = configs.num_hidden
        self.num_layers = configs.num_layers
        networks_map = {
            'CPL_base': CPL_base.RNN,
            'CPL':CPL.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        if configs.model_name=='CPL':
            self.prediction_optimizer = Adam(itertools.chain(self.network.encoder.parameters(),self.network.decoder.parameters(),
            self.network.prior.parameters(),self.network.embed_class.parameters(),self.network.embed_data.parameters()), lr=configs.lr)
            self.recon_optimizer=Adam(itertools.chain(self.network.head_reconstructor.parameters(),self.network.prior_fp.parameters(),
            self.network.shared_encoder.parameters(),self.network.latent_encoder.parameters()),lr=configs.lr)
        else:
            self.optimizer=Adam(self.network.parameters(),lr=configs.lr)
        self.MSE_criterion = nn.MSELoss( )

    def save(self, itr, prefix = ''):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, prefix + 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask,category):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()


        next_frames,loss,loss_pd,loss_kl = self.network(frames_tensor, mask_tensor,category)
        
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy(),loss_pd.detach().cpu().numpy(),loss_kl.detach().cpu().numpy()
    def CPL_train(self, pre_model,frames, mask,category,itr,is_replay=False):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        #do replay
        if is_replay:
            if itr%self.configs.replay_interval==0:
                #use generated image
                pre_model.network.eval()
                y=torch.eye(self.configs.num_classes)[category].to(self.configs.device)
                y=y.repeat(self.configs.batch_size,1)
                mu_prior, logvar_prior=pre_model.network.prior_fp(y)
                z_decode = pre_model.network.reparameterize(mu_prior, logvar_prior)
                z_decode=torch.cat([z_decode, y], dim = 1)
                gen_first_frame=pre_model.network.head_reconstructor(z_decode)
                gen_first_frame=gen_first_frame.unsqueeze(1)
                gen_first_frame=gen_first_frame.permute(0, 1, 3, 4, 2).contiguous()
                a=gen_first_frame.reshape(self.configs.batch_size,1,self.configs.img_width//self.configs.patch_size, self.configs.patch_size,
                            self.configs.img_width//self.configs.patch_size, self.configs.patch_size,
                            1)
                a=a.permute(0,1,2,4,3,5,6).contiguous()
                gen_first_frame=a.reshape(self.configs.batch_size,1,self.configs.img_width//self.configs.patch_size,self.configs.img_width//self.configs.patch_size,self.configs.patch_size*self.configs.patch_size)
                next_frames,_ = pre_model.network(gen_first_frame, mask_tensor,category,is_train=False,is_replay=True)
                gen_frames_tensor=torch.cat((gen_first_frame,next_frames),1)

                              
                if itr%(self.configs.replay_interval*1000)==0:
                    replay_frames=gen_frames_tensor.detach().cpu().numpy()
                    replay_frames = preprocess.reshape_patch_back(replay_frames, self.configs.patch_size)
                    res_path = os.path.join(self.configs.gen_frm_dir, 'replay_cat_'+str(category))
                    if not os.path.exists(res_path):
                        os.mkdir(res_path)
                    res_path = os.path.join(res_path, 'replay_itr_'+str(itr))
                    if not os.path.exists(res_path):
                        os.mkdir(res_path)
                    for i in range(replay_frames.shape[1]):
                            if i < 8:
                                name = 'pd0' + str(i + 2) + '.png'
                            else:
                                name = 'pd' + str(i + 2) + '.png'
                            file_name = os.path.join(res_path, name)
                            img_pd = replay_frames[0,i]
                            img_pd = np.maximum(img_pd, 0)
                            img_pd = np.minimum(img_pd, 1)
                            img_pd = np.uint8(img_pd * 255)
                            cv2.imwrite(file_name, img_pd)
                
                
                self.prediction_optimizer.zero_grad()
                self.recon_optimizer.zero_grad()
                next_frames,loss,loss_pd,loss_kl,loss_cat,loss_recon = self.network(gen_frames_tensor, mask_tensor,category,is_train=True,is_replay=False)
                loss.backward()
                self.prediction_optimizer.step()
                self.recon_optimizer.step()

        else:
            self.prediction_optimizer.zero_grad()
            self.recon_optimizer.zero_grad()
            next_frames,loss,loss_pd,loss_kl,loss_cat,loss_recon = self.network(frames_tensor, mask_tensor,category,is_train=True)
            loss.backward()
            self.prediction_optimizer.step()

            self.recon_optimizer.step()
        return loss.detach().cpu().numpy(),loss_pd.detach().cpu().numpy(),loss_kl.detach().cpu().numpy(),loss_cat.detach().cpu().numpy(),loss_recon.detach().cpu().numpy()
    

    def test(self, frames, mask,category):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        frames_tensor=frames_tensor.repeat(self.configs.num_samples,1,1,1,1)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        if self.configs.model_name=='CPL_base':
            right_cat_num=torch.tensor(0)
            next_frames= self.network(frames_tensor, mask_tensor,category,is_train=False)[0]

            return next_frames.detach().cpu().numpy()
        elif self.configs.model_name=='CPL':
            if self.configs.is_training:
                with torch.no_grad():
                    next_frames = self.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,0,is_train=0,is_replay=False)[0]
                    next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                    loss=nn.MSELoss()(next_frames[:,:self.configs.input_length],frames_tensor[:,:self.configs.input_length])
                    pred_cat=0
                    
                    for i in range(1,self.configs.num_classes):
                        next_frames = self.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,i,is_train=0,is_replay=False)[0]
                        next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                        loss_i=nn.MSELoss()(next_frames[:,:self.configs.input_length],frames_tensor[:,:self.configs.input_length])
                        if loss_i<loss:
                            loss=loss_i
                            pred_cat=i
                    next_frames,right_cat_num= self.network(frames_tensor, mask_tensor,pred_cat,is_train=0)
            else:
                #test_model.load(self.configs.pretrained_model)
                with torch.no_grad():
                    next_frames = self.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,0,is_train=0,is_replay=False)[0]
                    next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                    loss=nn.MSELoss()(next_frames,frames_tensor[:,:self.configs.input_length])
                    pred_cat=0       
                    for i in range(1,self.configs.num_classes):
                        next_frames = self.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,i,is_train=0,is_replay=False)[0]
                        next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                        loss_i=nn.MSELoss()(next_frames,frames_tensor[:,:self.configs.input_length])
                        if loss_i<loss:
                            loss=loss_i
                            pred_cat=i
                        
                if self.configs.is_TestAdapt:
                    ori_params = copy.deepcopy(self.state_dict())
                    for i in range(5):
                        self.prediction_optimizer.zero_grad()
                        next_frames, _ = self.network(frames_tensor, mask_tensor,pred_cat,is_train=0,is_replay=False)
                        next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                        loss=nn.MSELoss()(next_frames,frames_tensor[:,:self.configs.input_length])
                        loss.backward()
                        self.prediction_optimizer.step()
                    next_frames,right_cat_num= self.network(frames_tensor, mask_tensor,pred_cat,is_train=0)
                    self.load_state_dict(ori_params)
                    return next_frames.detach().cpu().numpy(),right_cat_num.detach().cpu().numpy()
                
                next_frames,right_cat_num= self.network(frames_tensor, mask_tensor,pred_cat,is_train=0)            
            # if pred_cat==category:
            #     right_cat_num+=self.configs.batch_size*self.configs.num_samples
            return next_frames.detach().cpu().numpy(),right_cat_num.detach().cpu().numpy()

    def parameters(self):
        return self.network.parameters()

    def named_parameters(self):
        return self.network.named_parameters()

    def load_state_dict(self, para):
        return self.network.load_state_dict(para)

    def state_dict(self):
        return self.network.state_dict()

