import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_fid.fid_score import calculate_fid_given_paths

from model import Generator, Discriminator
from utils import D_train, G_train, save_models
import os
from torchvision.utils import save_image

def save_real_samples(train_loader):
    real_images_dir = 'data/MNIST_raw'
    os.makedirs(real_images_dir, exist_ok=True)
    for batch_idx, (x, _) in enumerate(train_loader):
        if x.shape[0] != args.batch_size:
            image = x.reshape(x.shape[0],28,28)
        else:
            image = x.reshape(args.batch_size, 28, 28)
        for k in range(x.shape[0]):
            filename = os.path.join(real_images_dir, f'real_image_{batch_idx * args.batch_size + k}.png')
            save_image(image[k:k+1], filename)

# Function to generate fake samples using the generator
def generate_fake_samples(generator, num_samples):
    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).cuda()
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()


    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')
    if not(os.path.exists('data/MNIST_raw')):
        print('Saving test set locally ...')
        save_real_samples(test_loader)
    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.BCELoss()

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    print('Start Training :')
    
    n_epoch = args.epochs
    n_generator = 5
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            D_train(x, G, D, D_optimizer, criterion)
            if epoch % n_generator == 0:
            	G_train(x, G, D, G_optimizer, criterion)


        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')
        if epoch % 30 == 0:
            real_images_path = 'data/MNIST_raw'
            generated_images_path = 'samples'
            fid_value = calculate_fid_given_paths([real_images_path, generated_images_path],batch_size = 64,device = 'cuda',dims = 2048)
            print(f'Epoch {epoch}, FID: {fid_value:.2f}')    
    print('Training done')

        
