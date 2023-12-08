from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def train_epoch(cae, device, dataloader, loss_fn, optimizer):
    """The training loop of autoencoder.
    
    Args:
        cae (classes.resnet_autoencoder.AE): the autoencoder model with - by default- random initilized weights.
        device (str): if exists, the accelarator device used from the machine and supported from the pytorch else cpu.
        dataloader (DataLoader): loader with the training data.
        loss_fn (torch.nn.modules.loss): the loss function of the autoencoder
        optimizer (torch.optim): the optimizer of the autoencoder 

    Returns:
        (float): the mean of training loss
    """
    # Set train mode for both the encoder and the decoder
    cae.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for _, (x_batch, y_batch) in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)): # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        x_batch = x_batch.to(device)
        # CAE data
        decoded_batch,_ = cae(x_batch)
        # Evaluate loss
        loss = loss_fn(decoded_batch, x_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(cae, device, dataloader, loss_fn):
    """The validation loop of autoencoder on the test dataset.
    
    Args:
        cae (classes.resnet_autoencoder.AE): the autoencoder model.
        device (str): if exists, the accelarator device used from the machine and supported from the pytorch else cpu.
        dataloader (DataLoader): loader with the test data.
        loss_fn (torch.nn.modules.loss): the loss function of the autoencoder.

    Returns:
        (float): the validation loss.
    """
    # Set evaluation mode for encoder and decoder
    cae.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        decoded_data = []
        original_data = []
        for x_batch, _ in dataloader:
            # Move tensor to the proper device
            x_batch = x_batch.to(device)
            # CAE data
            decoded_batch,_ = cae(x_batch)
            # Append the network output and the original image to the lists
            decoded_data.append(decoded_batch.cpu())
            original_data.append(x_batch.cpu())
        # Create a single tensor with all the values in the lists
        decoded_data = torch.cat(decoded_data)
        original_data = torch.cat(original_data)
        # Evaluate global loss
        val_loss = loss_fn(decoded_data, original_data)

    return val_loss.data


def plot_ae_outputs(cae, dataset_opt, epoch, dataset, device, n=10):
    """Saving plot diagrams with reconstructed images in comparision with the original ones for a visual assessment.

    Args:
        cae (classes.resnet_autoencoder.AE): the trained autoencoder model.
        dataset_opt (str): the name on the input dataset. Proposed choices ['train_dataset', 'test_dataset']
        epoch (int): the present epoch in progress.
        device (str): if exists, the accelarator device used from the machine and supported from the pytorch else cpu.
        n (int): the number of original images to be plotted with their reconstructions. 10 (default).        
    """
    plt.figure(figsize=(16,4.5))
    targets = np.array(dataset.targets)
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):

        ax = plt.subplot(2,n,i+1)
        img = dataset[t_idx[i]][0] # dataset[t_idx[i]]-> tuple (X,Y)
        plt.imshow(img.permute((1, 2, 0))) # rgb
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Original images from ' + dataset_opt + ' epoch=' + str(epoch))

        ax = plt.subplot(2, n, i + 1 + n)
        img = img.unsqueeze(0).to(device) # img -> (3, xx, xx) but img.unsqueeze(0) -> (1,3,xx,xx)
        cae.eval()
        with torch.no_grad():
            rec_img, _  = cae(img)
        rec_img = rec_img.cpu().squeeze() # rec_img -> (1, 3, xx, xx) but img.squeeze() -> (3,xx,xx)
        plt.imshow(rec_img.permute((1, 2, 0))) # rgb
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Reconstructed images from ' + dataset_opt + ' epoch=' + str(epoch))

    if not os.path.isdir('output'):
        os.mkdir('output')
    # plt.show()
    plt.savefig(f'output/{epoch}_epoch_from_{dataset_opt}.png')


def checkpoint(model, epoch, val_loss, filename):
    """Saving the model at a specific state.

    Args:
        model (classes.resnet_autoencoder.AE): the trained autoencoder model.
        epoch (int): the present epoch in progress.
        val_loss (float): the validation loss.
        filename (str): the relative path of the file where the model will be stored.
    """
    torch.save(model.state_dict(), filename)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            }, filename)

def resume(model, filename):
    """Load the trained autoencoder model.

    Args:
        model (classes.resnet_autoencoder.AE): the untrained autoencoder model.
        filename (str): the relative path of the file where the model is stored.

    Results:
        model (classes.resnet_autoencoder.AE): the loaded autoencoder model.
        epoch (int): the last epoch of the training procedure of the model.
        loss (float): the validation loss of the last epoch.
    """
    checkpoint = torch.load(filename)
    model = model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    return model, epoch, loss