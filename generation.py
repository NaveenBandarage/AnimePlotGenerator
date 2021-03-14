def inference(model,input_text,device,top_k=5,length = 100):
    output = ''
    model.eval()
    tokens = config.tokenizer(input_text)
        
    h,c = model.zero_state(1)
    h = h.to(device)
    c = c.to(device)
    
    for t in tokens:
        output = output+t+' '
        pred,(h,c) = model(torch.tensor(w2i[t.lower()]).view(1,-1).to(device),(h,c))
        #print(pred.shape)
    for i in range(length):
        _,top_ix = torch.topk(pred[0],k = top_k)
               
        choices = top_ix[0].tolist()                
        choice = np.random.choice(choices)
        out = i2w[choice]
        output = output + out + ' '
        pred,(h,c) = model(torch.tensor(choice,dtype=torch.long).view(1,-1).to(device),(h,c))
    return output
  
  
  # ============================================================================================================
  
  
device = 'cpu'
mod = LSTMModel(emb_dim=config.emb_dim,hid_dim=config.hidden_dim,vocab_size=vocab_size,num_layers=3).to(device)
mod.load_state_dict(torch.load(config.model_path))
print('AI generated Anime synopsis:')
inference(model = mod, input_text = 'In the ', top_k = 30, length = 100, device = device)