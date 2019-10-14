from nn_policy import Policy, Oracle
from nn_discriminator import Discriminator

import constants as const

import math
import torch
import generator
import discriminator
import loader
import log


def device()

    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def generator_training(policy, rollout, oracle, optimizer, num_steps, seq_length, batchsize, mc_trials):

    policy.train()
    rollout.eval()

    rollout.set_parameters_to(policy)

    for _ in range(num_steps):
        batch, hidden = policy.inital()

        for length in seq_length:
            batch, hidden = generator.step(policy, batch, hidden)
            q_values = torch.empty([batchsize, 0])

            for _ in range(mc_trials):
                samples = generator.rollout(rollout, batch, hidden)
                samples = loader.load_single_batch(samples)
                reward = discriminator.evaluate_single_batch(samples)

                q_values = torch.cat([q_values, reward], dim=1)

            q_values = torch.mean(q_values, dim=1)
            generator.reward(q_values)

        generator.update(policy, optimizer)


def discriminator_training(model, generator, oracle, optimizer, criterion, num_steps, seq_length):

    model.train()
    generator.eval()

    for _ in range(num_steps):

        synthetic = generator.sample(generator, 1, seq_length)
        loader = loader.get_pos_neg_loader(synthetic)
        discriminator.update(model, optimizer, criterion, loader)


def adversarial_training():

    # initialization

    device = device()

    nn_discriminator = Discriminator()
    nn_policy = Policy()
    nn_rollout = Policy()
    nn_oracle = None

    if const.ORACLE:
        nn_oracle = Oracle()

    d_lr = const.DISCRIMINATOR_LEARNRATE
    g_lr = const.GENERATOR_LEARNRATE

    d_opt = torch.optim.Adam(nn_discriminator.parameters(), lr=d_lr)
    d_crit = torch.nn.BCELoss()
    p_opt = torch.optim.Adam(nn_policy.parameters(), lr=g_lr)

    iterations = const.ADVERSARIAL_ITERATIONS
    g_steps = const.ADVERSARIAL_GENERATOR_STEPS
    d_steps = const.ADVERSARIAL_DISCRIMINATOR_STEPS
    seq_length = const.ADVERSARIAL_SEQUENCE_LENGTH
    mc_trials = const.ADVERSARIAL_MONTECARLO_TRIALS
    batchsize = const.ADVERSARIAL_BATCHSIZE

    # start adversarial training

    loader.initialize(device)
    log.start_experiment()

    for i in range(iterations):

        discriminator_training(nn_discriminator, nn_rollout, nn_oracle, d_opt, d_crit, d_steps, seq_length)
        generator_training(nn_policy, nn_rollout, nn_oracle, g_steps, seq_length, batchsize, mc_trials)

        log.log(i, g_steps, d_steps, nn_policy, nn_discriminator, nn_oracle)

    # finish experiment

    directory = loader.get_experiment_directory()
    log.finish_experiment(directory)

    evaluation = generator.sample(policy, math.ceil(100 / batchsize), seq_length)
    loader.save_pngs(evaluation, directory + '/pngs')
