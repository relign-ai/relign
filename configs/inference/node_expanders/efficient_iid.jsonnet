{
  type: "efficient_iid_expander",
  branch_factor_strategy: {
    type: "list_branch_factor",
    branch_factors: [{"depth": 0, "branch_factor": 2}],
  },
  program: "{{prefix}}{{gen \"chain_of_thought\" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text=\"stop_text\" stop={stop} n={num_samples}}}",
  program_kwargs: {
    temperature: 0.6,
    max_tokens: 1024,
    top_p: 0.9,
    stop: '"\n\n\nProblem:"',
  },
  node_text_template: "{chain_of_thought}",
  model_context_size: 2047,
}
