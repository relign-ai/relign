local base_eval = import './base_eval.jsonnet';
local task_performance = import './analyzers/task_performance.jsonnet';

base_eval + {
  // Override any base settings if needed
  analysers: [
    task_performance,
  ],
}