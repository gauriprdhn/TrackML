"essential packages"
import tqdm
import torch
import numpy as np
import multiprocessing as mp
import random
import pandas
"libraries for debugging"
import sys
import os.path as osp

"custom imports"
from dataloader import fetch_dataloader
from gravnet_model import GravnetModel
from optimized_oc import calc_LV_Lbeta
import config 

loss_offset: float = 1.0

def compute_oc_loss(out, data, s_c=1., return_components=False):
    device = out.device
    pred_betas = torch.sigmoid(out[:,0])
    pred_cluster_space_coords = out[:,1:4]
    # pred_cluster_properties = out[:,3:]
    assert all(t.device == device for t in [
        pred_betas, pred_cluster_space_coords, data.y,
        data.batch,
        # pred_cluster_properties, data.truth_cluster_props
        ])
    out_oc = calc_LV_Lbeta(
        pred_betas,
        pred_cluster_space_coords,
        data.y.long(),
        data.batch,
        return_components=return_components
        )
    if return_components:
        return out_oc
    else:
        LV, Lbeta = out_oc
        return LV + Lbeta + loss_offset

def train(data_loader, model, epoch, optimizer,device):
    print('Training epoch', epoch)
    model.train()
    data = tqdm.tqdm(data_loader, total=len(data_loader))
    data.set_postfix({'loss': '?'})
    for i,inputs in enumerate(data):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        result = model(inputs.x, inputs.batch)
#        print("CHECK-1",result.device)
        loss = compute_oc_loss(result,inputs)
        print(f'loss={float(loss)}')
        loss.backward()
        optimizer.step()
        data.set_postfix({'loss': float(loss)})
        
    return model

def test(data_loader, model, epoch, device):
    with torch.no_grad():
        model.eval()
        loss = 0.
        for inputs in tqdm.tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(device)
            result = model(inputs.x, inputs.batch)
            loss +=  compute_oc_loss(result,inputs)
        loss /= len(data_loader)
        print(f'Avg test loss: {loss}')

def main():
    data = fetch_dataloader(data_dir = config.root,
                            batch_size = config.batch_size,
                            validation_split=config.validation_split,
                            full_dataset = False,
                            n_events = config.n_events,
                            pt_min = config.pt_min,
                            shuffle=config.shuffle)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader,test_loader = data['train'],data['val']
    model = GravnetModel(input_dim=config.input_dim, output_dim=config.output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.decay_rate)
  
    for i_epoch in range(config.epochs):
        model = train(train_loader, 
                      model, 
                      i_epoch, 
                      optimizer,
		      device)
        test(test_loader, 
             model, 
             i_epoch,
             device)
if __name__ == "__main__":
    main()
