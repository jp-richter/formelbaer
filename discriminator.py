
# speichere im datenset labels als float
# sorge im datenset dafuer, dass alles auf device laeuft
# biete methoden in dataclasse die die daten entsprechend vorbereitet, get loader ()

def evaluate_single_batch(nn_discriminator, samples):

    # samples: (batchsize, height, width)

    samples = samples[:,None,:,:]
    rewards = nn_discriminator(samples)

    return rewards[:,0] # [:,0] P(x ~ arxiv / oracle)


def evaluate_multiple_batches(nn_discriminator, loader):

    rewards = []

    for images, _ in loader:

        images = images[:,None,:,:]
        reward = nn_discriminator(images)
        rewards.append(reward)

    return rewards


def update(nn_discriminator, d_opt, d_crit, loader):

    for images, labels in loader:

        images.to(loader.device())
        labels.to(loader.device())

        # add channel
        images = images[:,None,:,:]

        outputs = nn_discriminator(images)
        loss = d_crit(outputs[:,1], labels)

        # output[:,0] P(x ~ arxiv / oracle)
        # output[:,1] P(x ~ generator)

        nn_discriminator.running_loss += loss.item()

        loss.backward()
        d_opt.step()
        