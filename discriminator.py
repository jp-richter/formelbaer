import config as cfg
from PIL import Image


def evaluate_single_batch(nn_discriminator, samples):

    # samples: (batchsize, height, width)

    samples = samples[:,None,:,:]
    rewards = nn_discriminator(samples)

    return rewards[:,0][:,None] # [:,0] P(x ~ arxiv / oracle)


def evaluate_multiple_batches(nn_discriminator, loader):

    rewards = []

    for images, _ in loader:

        images = images[:,None,:,:]
        reward = nn_discriminator(images)
        rewards.append(reward)

    return rewards

count = 0
def update(nn_discriminator, d_opt, d_crit, loader):
    global count

    for images, labels in loader:

        count += 1

        images.to(cfg.app_cfg.device)
        labels.to(cfg.app_cfg.device)

        import torchvision.transforms.functional as F
        from PIL import Image
        a = F.to_pil_image(images[0])
        a.show()

        # add channel
        images = images[:,None,:,:]

        print('COUNT')
        print(count)
        if count == 5:
            print('HELOOO')
            nn_discriminator.help = True

        outputs = nn_discriminator(images)

        print('LABELS')
        for l in labels:
            print(l)

        print('OUTPUTS')
        for o in outputs[:,1]:
            print(o)

        loss = d_crit(outputs[:,1], labels.float())

        # output[:,0] P(x ~ arxiv / oracle)
        # output[:,1] P(x ~ generator)

        nn_discriminator.running_loss += loss.item()

        loss.backward()
        d_opt.step()
        