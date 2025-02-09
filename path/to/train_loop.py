def learn(self):
    ...
    for iteration in tqdm(range(self.num_iterations)):
        if self.distributed_state.is_local_main_process:
            logger.info(f"===== TOP OF ITERATION {iteration} =====")

        # 1) Debug logs around generating episodes:
        logger.info(f"Rank {self.distributed_state.process_index}: About to _generate_episodes() for iteration {iteration}")
        episodes = self._generate_episodes(iteration=iteration, current_policy_path=current_policy_path)
        logger.info(f"Rank {self.distributed_state.process_index}: Returned from _generate_episodes()")

        # 2) Trainer step
        current_policy_path = self.trainer.step(episodes=episodes)
        logger.info(f"Rank {self.distributed_state.process_index}: Done with trainer step iteration {iteration}")

        # 3) Evaluate
        if iteration % self.evaluation_freq == 0:
            if self.distributed_state.is_main_process:
                logger.info("Evaluating current policy...")
                self._evaluate(iteration=iteration, current_policy_path=current_policy_path)
        self.distributed_state.wait_for_everyone()

        # 4) Save tokenizer
        ...
        self.distributed_state.wait_for_everyone()

        if is_local_main_process:
            logger.info(f"Iteration {iteration} complete!") 