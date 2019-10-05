import generator
import discriminator
import constants
import torch
import tokens


its = constants.ITERATIONS
gsteps = constants.GSTEPS
dsteps = constants.DSTEPS
sqlength = constants.SEQ_LENGTH
mcarlo = constants.MONTECARLO


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

				rewards = None
				for _ in range(mcarlo):
					samples = generator.rollout(batch, h)
					# verwandel batch in liste von baeumen
					# feede die baumlatex formeln in converter
					# bekomme liste von files zurueck
					# diese muessen in den discriminator
					# ergebnis sind die rewards

				# rewards = torch.ones([batch.shape[0]])
		
				generator.feedback(rewards)

			generator.update_policy()


if __name__ == "__main__":
	main()
