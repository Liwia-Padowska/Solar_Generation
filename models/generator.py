import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(opt.series_length)),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], * self.opt.img_shape)
        return img

# Add the same input to the discriminator, alongside the image input.
    # That may involve having inputs at different layers, so you can combine CNN and fully-connect
    # layers more easily. Typically you would concatenate the class choice to the flattened last CNN layer,
    # and use this concatenated vector as input to the first fully-connected layer. But you could concat the
    # class data to any nn layer before the output.