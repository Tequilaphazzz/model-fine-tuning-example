# Fine-Tuning Models using UnsloTh

This project demonstrates the process of fine-tuning modern models using the UnsloTh library. The project is designed to optimize the performance of language models such as Llama, Vicuna, and others, with the aim of improving performance in specific tasks, for example, in the field of mental health support or conversational models.

## Key Project Features
- **Support for Modern Models:**
  - Llama, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen, TinyLlama, Vicuna, Open Hermes, and others.
  - Capability to work with 16-bit LoRA and 4-bit QLoRA for speed and resource saving.
- **Flexible Configuration:**
  - Ability to set the maximum sequence length (`max_seq_length`) with automatic scaling.
- **Performance Optimization:**
  - Phi-3 Medium/Mini runs twice as fast thanks to the latest updates.

## Data Preparation
The pre-processed [Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) dataset is used for training. This dataset is provided in a format compatible with the Hugging Face `datasets` library.

Code example for loading:
```python
from datasets import load_dataset

data = load_dataset("Amod/mental_health_counseling_conversations")
````

## Installation

1.  Install the UnsloTh library using the following commands:

    ```bash
    pip install unsloth
    # For the latest  version:
    pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    ```

2.  Ensure you have installed additional dependencies required for model operation.

## Model Configuration

### Adding LoRA for Weight Updates

LoRA (Low-Rank Adaptation) is used to update a portion of the model's weights, which reduces computational resource requirements and speeds up training.
Code example:

```python
from unsloth import ModelTrainer

trainer = ModelTrainer(model="llama",
                       dataset="Amod/mental_health_counseling_conversations",
                       method="lora")

trainer.train()
```

### Training Parameters

  - **Training Method:** LoRA or QLoRA.
  - **Model Parameters:** Ability to set the maximum sequence length and automatic optimization.

## Advantages

  - **Resource Saving:** Using LoRA and QLoRA methods allows for a significant reduction in the memory and time required for training.
  - **Scalability:** Support for multiple models and automatic parameter scaling.
  - **Ease of Use:** Simple integration with popular libraries such as Hugging Face Datasets.

## Possible Improvements

  - Add support for a larger number of models and optimization methods.
  - Improve documentation to simplify working with the library.
  - Integration with other tools for model performance analysis.

## Conclusion

This project is an excellent example of using modern tools for configuring language models.
