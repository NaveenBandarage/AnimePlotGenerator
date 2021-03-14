def loss_fn(predicted,target):
    loss = nn.CrossEntropyLoss()
    return loss(predicted,target)

#====================================================================================================================================


def train_fn(model,device,dataloader,optimizer):
    model.train()
    tk0 = tqdm(dataloader,position=0,leave=True,total = num_batches)
    train_loss = AverageMeter()  
    hid_state,cell_state = model.zero_state(config.batch_size)
    hid_state = hid_state.to(device)
    cell_state = cell_state.to(device)
    losses = []
    for inp,target in tk0:
                
        inp = torch.tensor(inp,dtype=torch.long).to(device)
        target = torch.tensor(target,dtype=torch.long).to(device)

        optimizer.zero_grad()        
        pred,(hid_state,cell_state) = model(inp,(hid_state,cell_state))
        #print(pred.transpose(1,2).shape)
        
        loss = loss_fn(pred.transpose(1,2),target)
        
        hid_state = hid_state.detach()
        cell_state = cell_state.detach()
        
        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2) # to avoid gradient explosion
        optimizer.step()
        
        train_loss.update(loss.detach().item())
        tk0.set_postfix(loss = train_loss.avg)
        losses.append(loss.detach().item())
    return np.mean(losses)

#====================================================================================================================================

def run():
    device = 'cuda'
    model = LSTMModel(vocab_size=vocab_size,emb_dim=config.emb_dim,hid_dim=config.hidden_dim,num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = 'min', patience=2, verbose=True, factor=0.5)
    epochs = config.epochs
    
    best_loss = 999
    for i in range(1,epochs+1):
        train_dataloader = create_batches(batch_size=config.batch_size,input_tok=input_tok,seq_len=config.seq_len,target_tok=target_tok)
        print('Epoch..',i)
        loss = train_fn(model,device,train_dataloader,optimizer)
        if loss<best_loss:
            best_loss = loss
            torch.save(model.state_dict(),config.model_path)
        scheduler.step(loss)
        torch.cuda.empty_cache()
    return model