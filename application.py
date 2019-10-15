from nn_policy import Policy, Oracle
from nn_discriminator import Discriminator

import config as cfg

import math
import torch
import generator
import discriminator
import loader
import log
import os


# TODO wieso kann ich negative rewards bekommen??
# TODO wieso benutzen seqgans cross entropy und keine verteilungs divergenz?


def generator_training(nn_policy, nn_rollout, nn_discriminator, nn_oracle, g_opt, o_crit):

    nn_policy.train()
    nn_rollout.eval()

    for _ in range(cfg.app_cfg.g_steps):
        batch, hidden = nn_policy.initial()

        for length in range(cfg.app_cfg.seq_length):
            batch, hidden = generator.step(nn_policy, batch, hidden, nn_oracle, o_crit)
            q_values = torch.empty([cfg.app_cfg.batchsize, 0])

            for _ in range(cfg.app_cfg.montecarlo_trials):
                samples = generator.rollout(nn_rollout, batch, hidden)
                samples = loader.load_single_batch(samples)
                reward = discriminator.evaluate_single_batch(nn_discriminator, samples)

                q_values = torch.cat([q_values, reward], dim=1)

            q_values = torch.mean(q_values, dim=1)
            generator.reward(nn_policy, q_values)

        generator.update(nn_policy, g_opt)


def discriminator_training(nn_discriminator, nn_generator, d_opt, d_crit):

    nn_discriminator.train()
    nn_generator.eval()

    for _ in range(cfg.app_cfg.d_steps):

        synthetic = generator.sample(nn_generator, 1)
        torch_loader = loader.get_pos_neg_loader(synthetic)
        discriminator.update(nn_discriminator, d_opt, d_crit, torch_loader)


def adversarial_training():

    # INITIALIZATION

    print('Loading data..')

    loader.make_directories()
    loader.initialize() # must be called first

    print('Data successfully loaded.')

    nn_discriminator = Discriminator()
    nn_policy = Policy()
    nn_rollout = Policy()
    nn_oracle = Oracle() 

    if cfg.app_cfg.oracle:
        nn_oracle.load(cfg.paths_cfg.oracle)

    d_opt = torch.optim.Adam(nn_discriminator.parameters(), lr=cfg.d_cfg.learnrate)
    d_crit = torch.nn.BCELoss()
    g_opt = torch.optim.Adam(nn_policy.parameters(), lr=cfg.g_cfg.learnrate)
    o_crit = torch.nn.KLDivLoss()

    # START ADVERSARIAL TRAINING

    print('Starting experiment.')

    log.start_experiment()

    for i in range(cfg.app_cfg.iterations):
        nn_rollout.set_parameters_to(nn_policy)

        discriminator_training(nn_discriminator, nn_rollout, d_opt, d_crit)
        generator_training(nn_policy, nn_rollout, nn_discriminator, nn_oracle, g_opt, o_crit)

        log.write(i+1, nn_policy, nn_discriminator, nn_oracle, printout=True)

    # FINISH EXPERIMENT AND WRITE LOGS

    print('Finishing experiment.')

    directory = loader.get_experiment_directory()
    nn_policy.save(directory + '/policy_net.pt')
    nn_discriminator.save(directory + '/discriminator_net.pt')
    nn_oracle.save(directory + '/oracle_net.pt')

    log.finish_experiment(directory)

    evaluation = generator.sample(nn_policy, math.ceil(100 / cfg.app_cfg.batchsize))
    os.makedirs(directory + '/pngs')
    loader.save_pngs(evaluation, directory + '/pngs')


if __name__ == '__main__':
      adversarial_training()  
