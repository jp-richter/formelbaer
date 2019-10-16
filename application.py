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


# 


def generator_training(nn_policy, nn_rollout, nn_discriminator, nn_oracle, g_opt, o_crit):


    log.log.info('GENERATOR START')

    nn_policy.train()
    nn_rollout.eval()

    for _ in range(cfg.app_cfg.g_steps):
        batch, hidden = nn_policy.initial()

        for length in range(cfg.app_cfg.seq_length):
            log.log.info('GENERATOR Step START')
            batch, hidden = generator.step(nn_policy, batch, hidden, nn_oracle, o_crit, save_prob=True)
            log.log.info('GENERATOR Step END')
            q_values = torch.empty([cfg.app_cfg.batchsize, 0])

            if batch.shape[1] < cfg.app_cfg.seq_length:

                for _ in range(cfg.app_cfg.montecarlo_trials):
                    samples = generator.rollout(nn_rollout, batch, hidden)
                    samples = loader.load_single_batch(samples)
                    reward = discriminator.evaluate_single_batch(nn_discriminator, samples)

                    q_values = torch.cat([q_values, reward], dim=1)

                q_values = torch.mean(q_values, dim=1)

            else:
                # calculate reward for last step without montecarlo approximation
                log.log.info('GENERATOR Load/Reward START')
                samples = loader.load_single_batch(batch)
                reward = discriminator.evaluate_single_batch(nn_discriminator, samples)
                log.log.info('GENERATOR Load/Reward END')
                q_values = torch.cat([q_values, reward], dim=1)

            # average the reward over 
            q_values = torch.mean(q_values, dim=1)
            generator.reward(nn_policy, q_values)


        log.log.info('GENERATOR Update Start')
        generator.update(nn_policy, g_opt)
        log.log.info('GENERATOR Update End')

    log.log.info('GENERATOR END')


def discriminator_training(nn_discriminator, nn_generator, d_opt, d_crit):

    log.log.info('DISCR START')

    nn_discriminator.train()
    nn_generator.eval()

    for _ in range(cfg.app_cfg.d_steps):

        log.log.info('DISCR sample START')
        synthetic = generator.sample(nn_generator, 1)
        log.log.info('DISCR sample END')
        log.log.info('DISCR load and update START')
        torch_loader = loader.get_pos_neg_loader(synthetic)
        discriminator.update(nn_discriminator, d_opt, d_crit, torch_loader)
        log.log.info('DISCR load and update END')

    log.log.info('DISCR END')


def adversarial_training():

    # INITIALIZATION

    log.start_loading_data()
    loader.initialize() # must be called first
    log.finish_loading_data()

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

    log.start_experiment()

    for i in range(cfg.app_cfg.iterations):
        nn_rollout.set_parameters_to(nn_policy)

        discriminator_training(nn_discriminator, nn_rollout, d_opt, d_crit)
        generator_training(nn_policy, nn_rollout, nn_discriminator, nn_oracle, g_opt, o_crit)

        log.write(i+1, nn_policy, nn_discriminator, nn_oracle, printout=True)

    # FINISH EXPERIMENT AND WRITE LOGS

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
