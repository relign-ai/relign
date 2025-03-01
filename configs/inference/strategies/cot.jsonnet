{
  type: "cot_inference_strategy",
  samples: 8,
  question_field: "query",
  max_concurrent_generations: 128,
  max_concurrent_programs: 256,
  answer_extractor: {
    type: "identity_answer_extractor",
    node_key_name: "text",
  },
  guidance_llm: {
    type: "openai_vllm",
    api_key: "EMPTY",
    max_calls_per_min: 1000000,
    caching: false,
    max_retries: 10,
  },
  max_depth: 2,
}