{
  type: "math_episode_generator",
  num_episodes_per_iteration: 512,
  dataset_num_samples_per_iteration: 64, // 512 / 8
  reasoning_step_delimiter: "",
  answer_prefix: "\n\n# Answer\n",
  append_bos_to_query: true,
  append_eos_to_response: true,
  dataset_shuffle_on_each_iteration: true,
  dataset_sample_with_replacement: true,
  max_sequence_length: 2048,
  max_question_length: 1512,
  fill_missing_episodes: true,
  vllm_gpu_memory_utilization: "auto",
  wait_until_memory_release: true,
  save_generations_every_n_iteration: 2,
}