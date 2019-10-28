import config

import torch
import generator
import discriminator
import loader
import log
import math


# TODO rausfinden wie ich gewichte voreinstelle, dass atomare symbole zu anfang unwahrscheinlicher werden
# TODO manuell sinnvolles orakel erstellen


def train_with_mle(nn_policy, nn_oracle, epochs, num_samples) -> None:
    """
    This function trains the polcicy net with maximum likelihood estimation with the oracle distribution being the
    training target. Tthe oracle will generate the given amount of samples for the policy to train on. For the training
    itsself the task of the policy is to predict the next tokens correctly given subsequences of the oracle samples.
    This training method dodges the problems of computational costs of converting sequences to the image format and
    might be useful to compare the performance of the adversarial training to the typical mle approach.

    :param nn_policy: The policy net to train.
    :param nn_oracle: An oracle net that represents the oracle distribution, of type generator.Oracle.
    :param epochs: The amount of epochs for the policy net to train on the oracle generated data.
    :param num_samples: The amount of samples to be generated by the oracle.
    """

    nn_policy.train()
    nn_oracle.eval()

    criterion = torch.nn.NLLLoss()

    num_batches = math.ceil(num_samples / config.general.batch_size)
    oracle_batches = generator.sample(nn_oracle, num_batches)

    for epoch in range(epochs):
        for batch in range(oracle_batches):

            for length in range(config.general.sequence_length):

                input, hidden = nn_policy.inital()

                if length > 0:
                    input = batch[:, :length, :]

                target = batch[:, length, :]
                output, _ = nn_policy(input, hidden)
                output = torch.log(output)

                loss = criterion(output, target)
                loss.backward()
                nn_policy.optimizer.step()
                nn_policy.running_loss += loss.item()


def train_with_kldiv(nn_policy, nn_oracle, epochs, num_samples) -> None:
    """
    This function trains the policy to model the oracle distribution with the KLDiv loss objective. It directly compares
    the policy output give a subsequence to the policy output given by the oracle. In contrast to MLE this makes use
    of all the information about the oracle distribution available but obviously 'cheats' since it pretends to know the
    exact distribution of real data. It might still be useful as a performance comparison.

    :param nn_policy: The policy net to train.
    :param nn_oracle: The oracle net to generate fake real samples from.
    :param epochs: The number of epochs to train the policy.
    :param num_samples: The amount of samples for the policy net to train on.
    """

    nn_policy.train()
    nn_oracle.eval()

    criterion = torch.nn.KLDivLoss(reduction='batchmean')  # mean averages over input features too

    num_batches = math.ceil(num_samples / config.general.batch_size)

    for epoch in range(epochs):
        for batch in range(num_batches):

            batch_policy, hidden_policy = nn_policy.initial()
            batch_oracle, hidden_oracle = nn_oracle.initial()

            for length in range(config.general.sequence_length):
                batch_policy, hidden_policy = generator.step(nn_policy, batch_policy, hidden_policy)
                batch_oracle, hidden_oracle = generator.step(nn_oracle, batch_oracle, hidden_oracle)

                log_probs = torch.log(batch_policy[:, -1, :])
                target = batch_oracle[:, -1, :]

                loss = criterion(log_probs, target)
                loss.backward()
                nn_policy.optimizer.step()
                nn_policy.running_loss += loss.item()


def adversarial_generator(nn_policy, nn_rollout, nn_discriminator, iteration) -> None:
    """
    The training loop of the generating policy net.

    :param iteration: The current iteration of the adversarial training for the logging module.
    :param nn_policy: The policy net that is the training target.
    :param nn_rollout: The rollout net that is used to complete unfinished sequences for estimation of rewards.
    :param nn_discriminator: The CNN that estimates the probability that the generated sequences represent the real
        data distribution, which serve as reward for the policy gradient training of the policy net.
    :param nn_oracle: A policy net which gets initialized with high variance parameters and serves as fake real
        distribution to analyze the performance of the model even when no comparisons to other models can be made.
    """

    nn_rollout.set_parameters_to(nn_policy)
    nn_policy.train()
    nn_rollout.eval()
    nn_discriminator.eval()

    sequence_length = config.general.sequence_length
    montecarlo_trials = config.general.montecarlo_trialsa
    batch_size = config.general.batch_size

    def collect_reward(batch) -> torch.Tensor:

        images = loader.prepare_batch(batch)
        output = nn_discriminator(images)
        rewards = torch.empty(output.size(), device=config.general.device)

        for i in range(output.shape[0]):
            rewards[i][0] = 1 - output[i][0]

        return rewards

    batch, hidden = nn_policy.initial()

    for length in range(sequence_length):

        # generate a single next token given the sequences generated so far
        batch, hidden = generator.step(nn_policy, batch, hidden, save_prob=True)
        q_values = torch.empty([batch_size, 0], device=config.general.device)
        finished_sequence = batch.shape[1] < sequence_length

        # compute the Q(token,subsequence) values with monte carlo approximation
        if not finished_sequence:
            for _ in range(montecarlo_trials):
                samples = generator.rollout(nn_rollout, batch, hidden)
                reward = collect_reward(samples)
                q_values = torch.cat([q_values, reward], dim=1)
        else:
            reward = collect_reward(batch)
            q_values = torch.cat([q_values, reward], dim=1)

        # average the reward over all trials
        q_values = torch.mean(q_values, dim=1)
        nn_policy.rewards.append(q_values)

    generator.policy_gradient_update(nn_policy)
    log.generator_reward(nn_policy, iteration)


def adversarial_discriminator(nn_discriminator, nn_generator, nn_oracle, d_steps, d_epochs, epoch) -> None:
    """
    The training loop of the discriminator net.

    :param epoch: The current iteration of the adversarial training for the logging module.
    :param d_steps: The amount of steps the discriminator should be trained in one adversarial cycle.
    :param nn_generator: The policy net which generates the synthetic data the CNN gets trained to classify.
    :param nn_discriminator: The CNN that outputs an estimation of the probability that a given data point was generated
        by the policy network.
    :param nn_oracle: If the script uses oracle training fake real samples will be generated by the oracle net.
    :param d_epochs: The amount of epochs the discriminator trains per d step. In case of oracle training a samplesize
        can be specified and one epoch will contain the samplesize of positive and an equal amount of negative samples.
        In case the discriminator gets trained on arxiv data an upper limit of real samples can be specified and one
        epoch will contain the limit of real and an equal amount of generated samples.
    """

    nn_discriminator.train()
    nn_generator.eval()

    num_samples = config.general.size_real_dataset * 2 * d_steps  # equal amount of generated data
    data_loader = loader.prepare_loader(num_samples, nn_generator, nn_oracle)

    for d_epoch in range(d_epochs):
        for images, labels in data_loader:
            images = images.to(config.general.device)
            labels = labels.to(config.general.device)

            nn_discriminator.optimizer.zero_grad()
            outputs = nn_discriminator(images)

            # output[:,0] P(x ~ real)
            # output[:,1] P(x ~ synthetic)

            loss = nn_discriminator.criterion(outputs, labels.unsqueeze(dim=1).float())
            loss.backward()
            nn_discriminator.optimizer.step()
            nn_discriminator.running_loss += loss.item()
            nn_discriminator.running_acc += torch.sum((outputs[:, 0] > 0.5) == (labels == 1)).item()

        log.discriminator_loss(nn_discriminator, epoch, d_epoch)


def training() -> None:
    """
    The main loop of the script. To change parameters of the adversarial training parameters should not be changed here.
    Overwrite the configuration variables in config.py instead and start the adversarial training again.
    """

    loader.initialize()

    nn_discriminator = discriminator.Discriminator().to(config.general.device)
    nn_policy = generator.Policy().to(config.general.device)
    nn_rollout = generator.Policy().to(config.general.device)
    nn_oracle = generator.Oracle().to(config.general.device)

    nn_discriminator.criterion = torch.nn.BCELoss()
    nn_oracle.criterion = torch.nn.NLLLoss()
    nn_discriminator.optimizer = torch.optim.Adam(nn_discriminator.parameters(), lr=config.discriminator.learnrate)
    nn_policy.optimizer = torch.optim.Adam(nn_policy.parameters(), lr=config.generator.learnrate)

    # start adversarial training
    d_steps = config.general.d_steps
    g_steps = config.general.g_steps
    a_epochs = config.general.iterations
    d_epochs = config.general.d_epochs

    for epoch in range(a_epochs):

        # train discriminator
        adversarial_discriminator(nn_discriminator, nn_policy, nn_oracle, d_steps, d_epochs, epoch)

        # train generator
        for _ in range(g_steps):
            adversarial_generator(nn_policy, nn_rollout, nn_discriminator, epoch)

    loader.finish(nn_policy, nn_discriminator, nn_oracle)


def application() -> None:
    training()
    loader.shutdown()


if __name__ == '__main__':
    application()
