import torch 
import torchvision
import os
import argparse


from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2,
                      help="The batch size to use for training.")
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    checkpoint = torch.load('checkpoints_off_GPU/my_models.pth')
    model = Generator(g_output_dim = mnist_dim)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['G_state_dict'].items()})

    #model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples_test', exist_ok=True)
    os.makedirs('samples_train', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<100:
            z = torch.randn(args.batch_size, 100)
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<100:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples_test', f'{n_samples}.png'))         
                    n_samples += 1

