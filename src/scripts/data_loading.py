from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import os

def get_resized_transform():
    """Create the appropriate transformation for the loading data by
    resizing input images to 64x64 and transforming them to torch tensors.
    """
    transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToTensor()
      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
    return transform

def get_normalized_transform():
    """Create the appropriate transformation for the loading data by 
    normalizing input images and transforming them to torch tensors.
    """
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
    return transform

def get_simple_transform():
    """Create the appropriate transformation for the loading data by
    transforming input images to torch tensors.
    """
    transform = transforms.Compose([
      transforms.ToTensor()
  ])
    return transform

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    """Load cifar10 datasets.

    Args:
        download (bool): if true download data, else use the already downloaded data
        shuffle (bool): parameter of DataLoader class fro train and test loader
        batch_size (int): the numbers of smaples to load per batch

    Returns:
        train_loader (DataLoader): loader with the training data
        test_loader (DataLoader): loader with the test data
        train_dataset (torchvision.datasets): train dataset to be used in plotting
        test_dataset (torchvision.datasets): test dataset to be used in plotting
    """
    # Define the transforms

    # transform = get_resized_transform()
    transform = get_simple_transform()
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    train_dataset = datasets.CIFAR10('./data', train=True, download=download,
                                    transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=10, drop_last=False, shuffle=True)

    test_dataset = datasets.CIFAR10('./data', train=False, download=download,
                                    transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=10, drop_last=False, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset


if __name__=='__main__':
    # Load cifar-10 data
    BATCH_SIZE = 32
    print('Downloading & loading data...')
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_data_loaders(download=True, batch_size=BATCH_SIZE)
    # Check data shapes
    for x_batch, y_batch in train_loader:
        print(x_batch.shape)
        print(y_batch.shape)
        break