from nn_policy import Policy, Oracle
from nn_discriminator import Discriminator

import constants as const

import math
import torch
import generator
import discriminator
import loader
import log


def generator_training(nn_policy, nn_rollout, nn_discriminator, nn_oracle, p_opt, o_crit num_steps, seq_length, batchsize, mc_trials):

    nn_policy.train()
    nn_rollout.eval()

    nn_rollout.set_parameters_to(nn_policy)

    for _ in range(num_steps):
        batch, hidden = nn_policy.inital()

        for length in seq_length:
            batch, hidden = generator.step(nn_policy, batch, hidden, nn_oracle, o_crit)
            q_values = torch.empty([batchsize, 0])

            for _ in range(mc_trials):
                samples = generator.rollout(nn_rollout, batch, hidden)
                samples = loader.load_single_batch(samples)
                reward = discriminator.evaluate_single_batch(nn_discriminator, samples)

                q_values = torch.cat([q_values, reward], dim=1)

            q_values = torch.mean(q_values, dim=1)
            generator.reward(q_values)

        generator.update(nn_policy, p_opt)


def discriminator_training(nn_discriminator, nn_generator, d_opt, d_crit, num_steps, seq_length):

    nn_discriminator.train()
    nn_generator.eval()

    for _ in range(num_steps):

        synthetic = generator.sample(nn_generator, 1, seq_length)
        loader = loader.get_pos_neg_loader(synthetic)
        discriminator.update(nn_discriminator, d_opt, d_crit, loader)


def adversarial_training():

    # INITIALIZATION

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    loader.initialize(device) # must be called first

    nn_discriminator = Discriminator()
    nn_policy = Policy()
    nn_rollout = Policy()
    nn_oracle = Oracle() 

    if const.ORACLE:
        nn_oracle.load(const.FILE_ORACLE)

    d_lr = const.DISCRIMINATOR_LEARNRATE
    g_lr = const.GENERATOR_LEARNRATE

    d_opt = torch.optim.Adam(nn_discriminator.parameters(), lr=d_lr)
    d_crit = torch.nn.BCELoss()
    p_opt = torch.optim.Adam(nn_policy.parameters(), lr=g_lr)
    o_crit = torch.nn.CrossEntropyLoss()

    iterations = const.ADVERSARIAL_ITERATIONS
    g_steps = const.ADVERSARIAL_GENERATOR_STEPS
    d_steps = const.ADVERSARIAL_DISCRIMINATOR_STEPS
    seq_length = const.ADVERSARIAL_SEQUENCE_LENGTH
    mc_trials = const.ADVERSARIAL_MONTECARLO_TRIALS
    batchsize = const.ADVERSARIAL_BATCHSIZE

    # START ADVERSARIAL TRAINING

    log.start_experiment()

    for i in range(iterations):

        discriminator_training(nn_discriminator, nn_rollout, d_opt, d_crit, d_steps, seq_length)
        generator_training(nn_policy, nn_rollout, nn_discriminator, nn_oracle, p_opt, o_crit, g_steps, seq_length, batchsize, mc_trials)

        log.log(i, g_steps, d_steps, nn_policy, nn_discriminator, nn_oracle, printout=True)

    # FINISH EXPERIMENT AND WRITE LOGS

    directory = loader.get_experiment_directory()
    nn_policy.save(directory + '/policy.pt')
    nn_discriminator.save(directory + '/discriminator.pt')
    nn_oracle.save(directory + '/oracle.pt')

    log.finish_experiment(directory)

    evaluation = generator.sample(policy, math.ceil(100 / batchsize), seq_length)
    loader.save_pngs(evaluation, directory + '/pngs')
