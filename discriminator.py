import config as cfg


def evaluate_single_batch(nn_discriminator, image_batch):

    # image_batch: (batchsize, height, width)
    rewards = nn_discriminator(image_batch)

    return rewards[:,0][:,None] # [:,0] P(x ~ arxiv / oracle)


def evaluate_multiple_batches(nn_discriminator, loader):

    rewards = []

    for images, _ in loader:
        reward = nn_discriminator(images)
        rewards.append(reward)

    return rewards


def update(nn_discriminator, d_opt, d_crit, loader):

    for images, labels in loader:

        images.to(cfg.app_cfg.device)
        labels.to(cfg.app_cfg.device)

        outputs = nn_discriminator(images)
        loss = d_crit(outputs[:,1], labels.float())

        # output[:,0] P(x ~ arxiv / oracle)
        # output[:,1] P(x ~ generator)

        nn_discriminator.running_loss += loss.item()

        loss.backward()
        d_opt.step()
        