import torchvision
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

import matplotlib.pyplot as plt


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
        while n_samples<num_samples:
            z = torch.randn(args.batch_size, 100).cuda()
            x = generator(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<num_samples:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples_train', f'{n_samples}.png'))         
                    n_samples += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=128, 
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
    CP = False
    if CP:
        G_optimizer = optim.RMSprop(G.parameters(), lr=5e-5)
        D_optimizer = optim.RMSprop(D.parameters(), lr=5e-5)
    else:
        G_optimizer = optim.Adam(G.parameters(), lr = args.lr, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(D.parameters(), lr = args.lr, betas=(0.5, 0.999))

    print('Start Training :')
    
    n_epoch = args.epochs
    fid_values = []
    D_loss = []
    G_loss = []
    n_generator = 3
    z_fixed = torch.randn(1, 100)
    os.makedirs('samples_per_epoch', exist_ok=True)
    os.makedirs('samples_per_epoch_random', exist_ok=True)
    for epoch in range(1, n_epoch+1): 
        dl= 0
        gl= 0
        bpe = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            bpe+=1
            x = x.view(-1, mnist_dim)
            dl += D_train(x, G, D, D_optimizer, criterion)
            if epoch % n_generator == 0:
                gl += G_train(x, G, D, G_optimizer, criterion)
        if epoch % n_generator == 0:
            G_loss.append(gl/bpe)
	    print(f'Epoch {epoch}, G Loss {gl/bpe:.2f}, D Loss {dl/bpe:.2f}')
        D_loss.append(dl/bpe)

        if epoch % n_generator == 0:
            z_r = torch.randn(1, 100) 
            x_r = G(z_r)
            x_fixed = G(z_fixed)
            x_r = x_r.reshape(1, 28, 28)
            x_fixed = x_fixed.reshape(1, 28, 28)
            torchvision.utils.save_image(x_fixed[0], os.path.join('samples_per_epoch', f'{epoch}.png'))             
            torchvision.utils.save_image(x_r[0], os.path.join('samples_per_epoch_random', f'{epoch}.png'))
        
        if epoch % 20 == 0:
            save_models(G, D, 'checkpoints')
        if epoch % 20  == 0:
            real_images_path = 'data/MNIST_raw'
            generated_images_path = 'samples_train'
            generate_fake_samples(G, 10000)
            fid_value = calculate_fid_given_paths([real_images_path, generated_images_path],batch_size = 128,device = 'cuda',dims = 2048)
            print(f'Epoch {epoch}, FID: {fid_value:.2f}')
            fid_values.append(fid_value)

    save_models(G, D, 'checkpoints')
    G = G.cpu()
    D = D.cpu()
    os.makedirs('checkpoints_off_GPU', exist_ok=True)
    torch.save({'G_state_dict': G.state_dict(), 'D_state_dict': D.state_dict()}, 'checkpoints_off_GPU/my_models.pth')

    fig, ax = plt.subplots()
    ax.plot(fid_values, marker='o', linestyle='-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('FID')
    ax.set_title('FID Over Epochs')
    plt.savefig('fid_plot.png')

    fig, ax = plt.subplots()
    ax.plot(G_loss, marker='o', linestyle='-')
    ax.set_xlabel('Epoch*3')
    ax.set_ylabel('Generator Loss')
    ax.set_title('Generator Loss over training')
    plt.savefig('genloss.png')

    fig, ax = plt.subplots()
    ax.plot(D_loss, marker='o', linestyle='-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Discriminator Loss')
    ax.set_title('Discriminator Loss over training')
    plt.savefig('disloss.png')

    print('Training done')

        
