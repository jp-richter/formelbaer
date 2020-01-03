from config import config
from discriminator import Discriminator
from helper import store, get_logger

import torch
import generator
import loader
import tree
import os


def store_results(loss, reward_without_log_prob, entropy, prediction, policy):
    store.get('List: Mean Losses Per Generator Step').append(loss)
    store.get('List: Mean Rewards Per Generator Step').append(reward_without_log_prob)
    store.get('List: Mean Entropies Per Generator Step').append(entropy)
    store.get('List: Mean Predictions Per Generator Step').append(prediction)

    mean_policies = store.get('List: Mean Policies Per Single Step')
    mean_policies = torch.mean(torch.stack(mean_policies, dim=0), dim=0).cpu().detach()
    store.get('List: Mean Policies Per Generator Step').append(mean_policies)

    # calculate tuples of (action_id, count, average probability, average reward)
    sampled_actions = store.get('List: Sampled Actions Per Single Step')
    log_probs = store.get('List: Log Probabilites Per Actions Of Single Step')
    rewards = store.get('List: Rewards Per Single Step')

    action_counts = {}
    action_probs = {}
    action_rewards = {}

    assert len(sampled_actions) == len(log_probs) == len(rewards)
    batchsize = sampled_actions[0].shape[0]

    for step in range(len(log_probs)):
        for sample_id in range(batchsize):
            action = sampled_actions[step][sample_id]
            log_prob = log_probs[step][sample_id]
            reward = rewards[step][sample_id]

            if action in action_probs.keys():
                action_probs[action].append(log_prob)
                action_rewards[action].append(reward)
                action_counts[action] += 1
            else:
                action_probs[action] = [log_prob]
                action_rewards[action] = [reward]
                action_counts[action] = 1

    for action in action_counts.keys():
        action_probs[action] = (sum(action_probs[action]) / len(action_probs[action])).item()
        action_rewards[action] = (sum(action_rewards[action]) / len(action_rewards[action])).item()

    tuples = {a.item(): (action_counts[a], action_probs[a], action_rewards[a]) for a in action_counts.keys()}
    store.get('List: Action Info Dicts').append(tuples)

    step = store.get('Policy Step')
    store.set('Policy Step', step + 1)

    if step % 10 == 0:
        policy.save('{}/policies/{}'.format(store.folder, step))


def policy_gradient(policy):
    policy.optimizer.zero_grad()

    # weight state action values by log probability of action
    total = torch.zeros(config.batch_size, device=config.device)
    reward_with_log_prob = torch.zeros(config.batch_size, device=config.device)
    reward_without_log_prob = torch.zeros(config.batch_size, device=config.device)

    log_probs = store.get('List: Log Probabilites Per Actions Of Single Step')
    rewards = store.get('List: Rewards Per Single Step')

    assert len(rewards) == len(log_probs)
    assert all(tensor.size() == (config.batch_size,) for tensor in log_probs)
    assert all(tensor.size() == (config.batch_size,) for tensor in rewards)

    for log_prob, reward in zip(log_probs, rewards):
        total = total + (reward - config.g_baseline)
        reward_with_log_prob = reward_with_log_prob + (log_prob * total)
        reward_without_log_prob = reward_without_log_prob + total

    # average over batchsize
    reward_without_log_prob = torch.sum(reward_without_log_prob) / config.batch_size
    reward_with_log_prob = torch.sum(reward_with_log_prob) / config.batch_size

    # negate for gradient descent and substract entropy
    entropies = store.get('List: Mean Entropies Per Single Step')
    entropy = 0.01 * sum(entropies) / len(entropies)

    loss = - (reward_with_log_prob + entropy)
    loss.backward()
    policy.optimizer.step()

    prediction = sum(rewards[-1]) / config.batch_size
    store_results(loss.item(), reward_without_log_prob.item(), entropy.item(), prediction.item(), policy)


def collect_reward(discriminator, batch):
    """
    This function calculates the rewards given a batch of onehot sequences with the given discriminator. The rewards
    will be the probability that the sequences are no synthetic predicted by the discriminator.

    :param discriminator: The discriminator which predictions the rewards are based on.
    :param batch: The batch of sequences generated by the generator.
    :return: Returns a tensor of size (batchsize, 1).
    """

    images = loader.prepare_batch(batch)
    output = discriminator(images)
    reward = torch.empty((batch.shape[0], 1), device=config.device)

    # TODO punish atomic expressions
    # TODO minimize negative output instead of maximizing 1-output !

    for r in range(output.shape[0]):
        # reward[r][0] = 1 - output[r]
        reward[r][0] = - output[r]

    return reward


def adversarial_generator(policy, rollout, discriminator, adversarial_step, g_steps):
    rollout.set_parameters_to(policy)
    policy.train()
    rollout.eval()
    discriminator.eval()

    # results of a training step
    store.set('List: Mean Losses Per Generator Step', [], attributes=[store.PLOTTABLE], if_exists=False)
    store.set('List: Mean Rewards Per Generator Step', [], attributes=[store.PLOTTABLE], if_exists=False)
    store.set('List: Mean Entropies Per Generator Step', [], attributes=[store.PLOTTABLE], if_exists=False)
    store.set('List: Mean Predictions Per Generator Step', [], attributes=[store.PLOTTABLE], if_exists=False)

    store.set('List: Action Info Dicts', [], if_exists=False)
    store.set('List: Mean Policies Per Generator Step', [], if_exists=False)
    store.set('List: Action Counts Per Generator Step', [], if_exists=False)
    store.set('List: Formular Examples', [], if_exists=False)

    for step in range(g_steps):

        print('Global Step {} - Generator Step {}'.format(adversarial_step, step))

        # temporary store - necessary for loss calculation - should be overwritten each step
        store.set('List: Log Probabilites Per Actions Of Single Step', [])
        store.set('List: Rewards Per Single Step', [])
        store.set('List: Mean Entropies Per Single Step', [])

        # temporary store - not necessary for loss calculation - should be overwritten each step
        store.set('List: Sampled Actions Per Single Step', [])
        store.set('List: Mean Policies Per Single Step', [])

        batch, hidden = policy.initial()

        for length in range(config.sequence_length):

            # generate a single next token given the sequences generated so far
            batch, hidden = generator.step(policy, batch, hidden, save_prob=True)
            q_values = torch.empty([config.batch_size, 0], device=config.device)

            # compute the Q(token,subsequence) values with monte carlo approximation
            if not batch.shape[1] < config.sequence_length:
                for _ in range(config.montecarlo_trials):
                    samples = generator.rollout(rollout, batch, hidden)
                    reward = collect_reward(discriminator, samples)
                    q_values = torch.cat([q_values, reward], dim=1)
            else:
                reward = collect_reward(discriminator, batch)
                q_values = torch.cat([q_values, reward], dim=1)

            # average the reward over all trials
            q_values = torch.mean(q_values, dim=1)
            store.get('List: Rewards Per Single Step').append(q_values)

            # generator.policy_gradient_update(policy)  # TODO comment out to reward like in SeqGAN
            # batch, hidden = (batch.detach(), hidden.detach())  # TODO comment out to reward like in SeqGAN

        store.get('List: Formular Examples').append(', '.join(tree.to_latex(batch[-3:].tolist())))
        policy_gradient(policy)


def adversarial_discriminator(discriminator, policy, adversarial_step, d_steps, d_epochs):
    discriminator.reset()
    discriminator.train()
    policy.eval()

    store.set('Discriminator Loss', [], attributes=[store.PLOTTABLE], if_exists=False)
    store.set('Discriminator Accuracy', [], attributes=[store.PLOTTABLE], if_exists=False)

    num_samples = config.num_real_samples * 2 * d_steps  # equal amount of generated data
    data_loader = loader.prepare_loader(num_samples, policy)

    for epoch in range(d_epochs):
        store.set('Discriminator Loss Per Batch', [])
        store.set('Discrmininator Accuracy Per Batch', [])

        print('Global Step {} - Discriminator Epoch {}'.format(adversarial_step, epoch))

        for images, labels in data_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)

            discriminator.optimizer.zero_grad()
            outputs = discriminator(images)

            # output[:,0] P(x ~ real)
            # output[:,1] P(x ~ synthetic)

            loss = discriminator.criterion(outputs, labels.float())
            loss.backward()
            discriminator.optimizer.step()

            store.get('Discriminator Loss Per Batch').append(loss.item())
            store.get('Discrmininator Accuracy Per Batch').append(torch.sum((outputs > 0.5) == (labels == 1)).item()
                                                                  / outputs.shape[0])

        loss = store.get('Discriminator Loss Per Batch')
        acc = store.get('Discrmininator Accuracy Per Batch')
        store.get('Discriminator Loss').append(sum(loss) / len(loss))
        store.get('Discriminator Accuracy').append(sum(acc) / len(acc))


def training(discriminator, policy, rollout):
    """
    The main loop of the script. To change parameters of the adversarial training parameters should not be changed here.
    Overwrite the configuration variables in config.py instead and start the adversarial training again.
    """

    print('Starting adversarial training..')
    for adversarial_step in range(config.adversarial_steps):

        adversarial_discriminator(discriminator, policy, adversarial_step, config.d_steps, config.d_epochs)
        adversarial_generator(policy, rollout, discriminator, adversarial_step, config.g_steps)

        if not adversarial_step == 0 and adversarial_step % 20 == 0 and config.num_real_samples < 10000:
            config.num_real_samples += 1000

    print('Finished training and saving results.')
    return discriminator, policy


def initialize():
    """
    Setup neural networks and tensorboard logging.
    """

    discriminator = Discriminator().to(config.device)
    policy = generator.Policy().to(config.device)
    rollout = generator.Policy().to(config.device)

    discriminator.criterion = torch.nn.BCELoss()
    discriminator.optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.d_learnrate)
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=config.g_learnrate)

    hyperparameter = {k: v for k, v in config.__dict__.items() if not v == config.device}

    # TODO Plan: 0.25 Spruenge machen und dann das ganze noch mal, mit der Modifikation in collect reward etc.
    # TODO: dabei auch dings wieder reinnehmen: entropie

    notes = '''
    0.1 Added Multipages   
    0.2 Resetting D Weights After Each Step
    0.3 Set Batchsize to cores
    0.4 Switched to update policy after each step in a sequence
    0.5 Switched learningrate from 0.05 to 0.001
    0.6 Switched back to update after sequence, batchsize to 4*cores
    0.7 Increment real samples D trains by 2000 Samples every 20 Epochs (max 10.000)
    0.8 Loss + Entropy * beta - Gamma 1 - Bias - switched to 1000 samples per epoch
    0.9 entropy beta 0.01 -> 0.005
    1.0 learnrate 0.005 -> 0.01
    1.1 learnrate 0.01 -> 0.02, removed entropy
    1.2 learnrate 0.02 -> 0.03
    1.3 learnrate 0.03 -> 0.04
    1.4 learnrate 0.04 -> 0.05
    1.5 learnrate 0.05 -> 0.075
    1.6 learnrate 0.075 -> 0.1
    1.7 learnrate 0.1 -> 0.5
    0.5 -> 1
    entropy wieder rein, objective umgedreht, lr 0.2, baseline 0.1, 
    entropyz von 0.005 auf 0.01, lr 0.1, 0.008, 0.06, 0.04, 0.02, 0.01
    '''

    store.setup(loader.make_directory_with_timestamp(), hyperparameter, notes)
    store.set('Policy Step', 0)
    os.makedirs('{}/policies'.format(store.folder))

    log = get_logger(store.folder, 'errors')

    return discriminator, policy, rollout,  log


def application():
    log = None

    try:
        loader.initialize()

        discriminator, policy, rollout, log = initialize()
        training(discriminator, policy, rollout)

        loader.finish(policy, discriminator)

    except Exception as e:
        print(str(e))
        log.error(str(e))


if __name__ == '__main__':
    application()
