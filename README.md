# Exploring the potential of Wasserstein GAN for image generation on MNIST

## Project guidelines
Under a fixed generator architecture, and simplicity constraints on the discriminator architecture, explore variosu techniques for improving image generation for Generative Adversarial Network. A team, composed of Augustin Gervreau, Thomas Boudras and myself, decided to explore the improvements that could be made by changing the loss function to a Wasserstein divergence.

## Explored techniques
- On branch **vanilla_data_augment**, we first explore main improvements to the vanilla GAN architecture for improving the FID of the model. This included adding batch normalization, dropout and implementing differentiable data augmentation

- On branch **LIP_clip** we first implement Wasserstein GAN with weight clipping. 
- On branch **LIP_grad_penalty**, we implement Wasserstein GAN with gradient penalty
- On **main** branch, we implement WGAN with gradient penalty and Convolutional networks added to the discriminator, which gave us the best FID score of 6.

## Files
[**train.py**](./train.py) : is used for training the model
[**utils.py**](./utils.py) : is used for managing the inner training loops of the discriminator and generator
[**generate.py**](./generate.py) : is used for generating images
[**model.py**](./model.py) : model architectures
[**Rapport_datalab.pdf**](./Rapport_datalab.pdf) : project report


