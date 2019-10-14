
# speichere im datenset labels als float
# sorge im datenset dafuer, dass alles auf device laeuft
# biete methoden in dataclasse die die daten entsprechend vorbereitet, get loader ()

def evaluate_single_batch(model, samples):

    # samples: (batchsize, height, width)

    samples = samples[:,None,:,:]
    rewards = model(samples)

    return rewards[:,0] # [:,0] P(x ~ arxiv)


def evaluate_multiple_batches(model, loader):

    rewards = []

    for images, _ in loader:

        images = images[:,None,:,:]
        reward = model(images)
        rewards.append(reward)

    return rewards


def update(model, optimizer, criterion, loader):

    for images, labels in loader:

        images.to(loader.device())
        labels.to(loader.device())

        # add channel
        images = images[:,None,:,:]

        outputs = model(images)
        loss = criterion(outputs[:,1], labels)

        # output[:,0] P(x ~ arxiv)
        # output[:,1] P(x ~ generator)

        model.running_loss += loss.item()

        loss.backward()
        optimizer.step()
        