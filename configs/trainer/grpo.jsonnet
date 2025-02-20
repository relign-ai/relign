{
    inference_strategy+ {
        node_expander+ {
            type: 'efficient_iid',
            program = $.prompt_library.tree.expansion.iid,
            program_kwrags: {
                temperature: 0.95,
                top_p: 0.9,
                top_k: 50,
                max_tokens: 256,
                stop_regex: '"\nStep"',
            },
            node_text_template: $.prompt_library.tree.question_template,
        }
    }
}