import generator
import rewards
import constants
import torch
import tokens
import tree
import converter


its = constants.ITERATIONS
gsteps = constants.GSTEPS
dsteps = constants.DSTEPS
sqlength = constants.SEQ_LENGTH
mcarlo = constants.MONTECARLO


def parse(batch):

    trees = []
    for sample in batch:
        sequence = []

        for onehot in sample:
            sequence.append(tokens.id(onehot))

        trees.append(tree.parse(sequence))

    return trees


def main():
    # pre training

    # adversarial training
    for iteration in range(its):

        for _ in range(dsteps):
            pass

        generator.update_rollout()

        for _ in range(gsteps):
            batch = h = None

            for length in range(sqlength):
                batch, h = generator.step(batch, h)
                reward = None

                for _ in range(mcarlo):
                    
                    samples = generator.rollout(batch, h)
                    trees = parse(samples)
                    latexs = [tree.latex() for tree in trees]
                    folder = converter.convert(latexs)

                    if reward is None:
                        reward = rewards.rewards(folder)
                    else:
                        reward = torch.cat([reward, rewards.rewards(folder)], dim=1)

                rewards = torch.mean(reward, dim=1)
                generator.feedback(rewards)

            generator.update_policy()


if __name__ == "__main__":
    main()
