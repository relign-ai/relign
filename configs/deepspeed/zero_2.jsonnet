(import 'base.jsonnet') + {
    zero_optimization: {
        stage: 2,
        allgather_partitions: true,
        allgather_bucket_size: 5e8,
        overlap_comm: false,
        reduce_scatter: true,
        reduce_bucket_size: 'auto',
        contiguous_gradients: true,
    },
}