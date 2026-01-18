"""
Variational Autoencoder (VAE) for MNIST.
Compresses 28x28 images into a low-dimensional latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Simple Variational Autoencoder with fully-connected layers.
    
    Architecture:
        Encoder: input -> hidden -> (mu, logvar)
        Decoder: latent -> hidden -> output
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16):
        """
        Args:
            input_dim: Flattened input dimension (28*28 for MNIST)
            hidden_dim: Size of hidden layer
            latent_dim: Dimensionality of latent space
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # ===== Encoder =====
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ===== Decoder =====
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = F.relu(self.encoder_hidden(x))
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean (batch_size, latent_dim)
            logvar: Log variance (batch_size, latent_dim)
            
        Returns:
            z: Sampled latent vector (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from N(0,1)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            
        Returns:
            reconstruction: Reconstructed input (batch_size, input_dim)
        """
        h = F.relu(self.decoder_hidden(z))
        return torch.sigmoid(self.decoder_out(h))
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            recon: Reconstruction (batch_size, input_dim)
            mu: Latent mean (batch_size, latent_dim)
            logvar: Latent log variance (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        recon: Reconstructed input (batch_size, input_dim)
        x: Original input (batch_size, input_dim)
        mu: Latent mean (batch_size, latent_dim)
        logvar: Latent log variance (batch_size, latent_dim)
        beta: Weight for KL term (beta-VAE)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (BCE)
        kl_loss: KL divergence
    """
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(
        recon, x, reduction='sum'
    )
    
    # KL divergence: KL(q(z|x) || p(z))
    # Closed form for Gaussian: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss