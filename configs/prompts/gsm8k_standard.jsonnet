{
  standard: std.stripChars(|||
    [MATH_TASK] Problem:
    {query}

    Solution:
  |||, "\n"),
  
  reasoning_wrapped: std.stripChars(|||
    [MATH_TASK] Problem:
    {query}
    first thinks about the reasoning process in and provides the the answer. The reasoning
    process and answer are enclosed within <think> </think> followed by an answer tag: \n####, respectively, i.e.,
    <think> reasoning process here </think> \n#### final answer
  |||, "\n"),
}