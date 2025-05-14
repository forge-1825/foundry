PIPELINE_CONFIG = {
    'steps': [
        {
            'id': 'manual_extractor',
            'name': 'Manual Extractor',
            'script': 'manual_extractor.py',
            'requires_gpu': False
        },
        {
            'id': 'data_enrichment',
            'name': 'Data Enrichment',
            'script': 'data_enrichment_enhanced_gpu_fixed_v2.py',
            'requires_gpu': True
        },
        {
            'id': 'teacher_pair_generation',
            'name': 'Teacher Pair Generation',
            'script': 'teacher_pair_generation_vllm_hierarchical.py',
            'requires_server': ['teacher']
        },
        {
            'id': 'distillation',
            'name': 'Distillation Training',
            'script': 'distillation_vllm_faster_improved.py',
            'requires_server': ['teacher', 'student'],
            'requires_gpu': True
        },
        {
            'id': 'student_self_study',
            'name': 'Student Self-Study',
            'script': 'student_self_study_enhanced.py',
            'requires_server': ['student'],
            'requires_gpu': True
        },
        {
            'id': 'merge_model',
            'name': 'Model Merging',
            'script': 'merge_model.py',
            'requires_gpu': False
        },
        {
            'id': 'evaluation',
            'name': 'Model Evaluation',
            'script': 'evaluate_distilled.py',
            'requires_server': ['teacher', 'student'],
            'requires_gpu': True
        }
    ],
    'default_params': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'max_length': 512
    }
}

