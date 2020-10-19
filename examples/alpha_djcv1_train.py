import sklearn
import numpy as np

from deepjet_geometric.datasets import DeepJetCoreV1
from torch_geometric.data import DataLoader
import os

data_train = DeepJetCoreV1("/data/t3home000/bmaier/tor/train/")
data_test = DeepJetCoreV1("/data/t3home000/bmaier/tor/test/")

train_loader = DataLoader(data_train, batch_size=600, shuffle=True,
                          follow_batch=['x_track', 'x_sv'])
test_loader = DataLoader(data_test, batch_size=600, shuffle=True,
                         follow_batch=['x_track', 'x_sv'])

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear

import utils

MAXSTEP = 10
BATCHSIZE = 600
OUTPUT = '/home/bmaier/public_html/figs/graphb/v1/'
model_dir = '/data/t3home000/bmaier/graphb/v0/'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 128
        
        self.sv_encode = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.trk_encode = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.conv_1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.ELU()
            ),
            k=8
        )

        self.conv_2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.ELU()            
            ),
            k=8
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 4),
            nn.ELU(),
            nn.Linear(4, 1)
            #nn.Sigmoid()
        )
        
    def forward(self, x_sv, x_trk, batch_sv, batch_trk):
        x_sv_enc = self.sv_encode(x_sv)
        x_trk_enc = self.trk_encode(x_trk)
        
        feats_1 = self.conv_1(x=(x_sv_enc, x_trk_enc), batch=(batch_sv, batch_trk))        
        feats_2 = self.conv_2(x=(feats_1, x_trk_enc), batch=(batch_trk, batch_trk))        

        out, batch = avg_pool_x(batch_trk, feats_2, batch_trk)
        out = self.output(out)

        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dummy = Net().to(device)
optimizer = torch.optim.Adam(dummy.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def train():
    dummy.train()

    total_loss = 0
    counter = 0

    hist_dict_s = {}
    hist_dict_b = {}

    for data in train_loader:
        counter += 1
        print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)))
        data = data.to(device)
        optimizer.zero_grad()
        out = dummy(data.x_sv,data.x_track,data.x_sv_batch,data.x_track_batch)
        '''
        for i in range(8):
            if "%i" %i not in hist_dict_s:
                tmp_hist = utils.NH1(np.arange(np.amin(data[data.y.bool()].x_track[:,i].detach().cpu().numpy()),np.amax(data[data.y.bool()].x_track[:,i].detach().cpu().numpy()),(np.amax(data[data.y.bool()].x_track[:,i].detach().cpu().numpy())-np.amin(data[data.y.bool()].x_track[:,i].detach().cpu().numpy()))/20))
                hist_dict_s["%i" %i] = tmp_hist
                hist_dict_b["%i" %i] = tmp_hist
                #print(~data.y)
                #print(data.y)
                
                hist_dict_s["%i" %i].fill_array(data[~data.y.bool()].x_track[:,i].detach().cpu().numpy())
                hist_dict_b["%i" %i].fill_array(data[data.y.bool()].x_track[:,i].detach().cpu().numpy())
            else:
                hist_dict_s["%i" %i].fill_array(data[~data.y.bool()].x_track[:,i].detach().cpu().numpy())
                hist_dict_b["%i" %i].fill_array(data[data.y.bool()].x_track[:,i].detach().cpu().numpy())
        '''
        #print(data.x_track)
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(out[0]), data.y.float())
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if counter*BATCHSIZE > 30000:
            print("Now returning")
            break

    #return total_loss / MAXSTEP*BATCHSIZE
    return total_loss / len(train_loader.dataset)#, hist_dict_s, hist_dict_b

def test(test_loader):
    dummy.eval()

    correct = 0
    counter = 0

    hists = utils.NH1(np.arange(-0.1,1.1,0.01))
    histb = utils.NH1(np.arange(-0.1,1.1,0.01))

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = dummy(data.x_sv,data.x_track,data.x_sv_batch,data.x_track_batch)#.max(dim=1)[1]
            pred = torch.sigmoid(pred[0])
            #print(data.x_track)

        hists.fill_array(pred[data.y.bool()].detach().cpu().numpy(), weights=None)
        histb.fill_array(pred[~data.y.bool()].detach().cpu().numpy(), weights=None)

        correct += torch.round(pred).eq(data.y).sum().item()
        counter += 1

        if counter*BATCHSIZE > 30000:
            break

    #return float(correct) / MAXSTEP*BATCHSIZE, hists, histb
    return correct / len(test_loader.dataset), hists, histb

for epoch in range(1, 50):
    loss = train()
    #loss, input_vars_s, input_vars_b = train()
    test_acc, hists, histb = test(test_loader)
    scheduler.step()

    p = utils.Plotter()
    hists.scale()
    histb.scale()
    p.clear()
    p.add_hist(hists, 'Hbb', 'indianred')
    p.add_hist(histb, 'QCD', 'cornflowerblue')
    p.plot(xlabel = 'Network score', ylabel = 'Arbitrary units', output=OUTPUT+'{0:0=3d}'.format(epoch)+'_graphb_score')
    
    '''
    for i in range(8):
        p.clear()
        input_vars_s["%i"%i].scale()
        input_vars_b["%i"%i].scale()
        p.add_hist(input_vars_s["%i"%i], 'Hbb', 'indianred')
        p.add_hist(input_vars_b["%i"%i], 'QCD', 'cornflowerblue')
        p.plot(xlabel = 'Network score', ylabel = 'Arbitrary units', output=OUTPUT+'{0:0=3d}'.format(epoch)+'_%i'%i)
    '''    

    hists_hbb = {'NN':hists}
    hists_qcd = {'NN':histb}

    r = utils.Roccer()
    r.clear()
    r.add_vars(hists_hbb,
               hists_qcd,
               {'NN':'Network score'},
               {'NN':'indianred'})
    r.plot(output=OUTPUT+'{0:0=3d}'.format(epoch)+'_graphb_roc')

    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))

    state_dicts = {'model':dummy.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))

