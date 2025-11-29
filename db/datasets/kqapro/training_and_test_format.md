# Training / Validation Set Format
```json

[
    {
        'question': str,
        'sparql': str, # executable in our virtuoso engine
        'program': 
        [
            {
                'function': str,  # function name
                'dependencies': [int],  # functional inputs, representing indices of the preceding functions
                'inputs': [str],  # textual inputs
            }
        ],
        'choices': [str],  # 10 answer choices
        'answer': str,  # golden answer
    }
]
```

# Test Set Format
```json

[
    {
        'question': str,
        'choices': [str],  # 10 answer choices
    }
]
```