```
tasks
├── task_evaluation/
│   └── mgsm/                 # MGSM
|       ├── mgsm_id.jsonl     # Indonesian version of MGSM
│       └── mgsm.py           # Handles MGSM task format and metrics
│   └── belebele.py           # Handles Belebele task format and metrics
│   └── global_mmlu.py        # Handles Global MMLU task format and metrics
│   └── mclm_mt_math100.py    # Handles MT MATH 100 task format and metrics
├── task_modifiers/                       
│   └── reasoning_prompt.py   # Stores prompts for reasoning in each language
│   └── translate_prompt.py   # Stores prompts for translation
└── task_training/
    └── deepscaler.py         # Handles training data
```
