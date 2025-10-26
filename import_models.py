"""
import_models.py

This script demonstrates how to load Google DeepMind's Gemma model and OpenAI's GPT-OSS model using Hugging Face's transformers library.

Note: Running these models requires significant compute resources. Adjust the model IDs and device settings as needed.
"""

from transformers import pipeline
import torch


def load_gemma(model_id: str = "google/gemma-3-4b-it", device: int = 0):
    """
    Load Gemma model using Hugging Face pipeline.

    :param model_id: Hugging Face model identifier for Gemma.
    :param device: Device to load the model on (e.g., 0 for first GPU, -1 for CPU).
    :return: A text-generation pipeline instance.
    
    The Gemma model is loaded as a text generation pipeline with the instruction-tuned variant.
    See the Gemma documentation for details on model loading and supported tasks【459652164890342†L368-L384】.
    """
    return pipeline(
        task="text-generation",
        model=model_id,
        device=device,
        torch_dtype=torch.bfloat16,
    )


def load_gpt_oss(model_id: str = "openai/gpt-oss-120b", device_map: str = "auto"):
    """
    Load GPT-OSS model using Hugging Face pipeline.

    :param model_id: Hugging Face model identifier for GPT-OSS.
    :param device_map: Device map for automatic placement across available devices.
    :return: A text-generation pipeline instance.

    The GPT-OSS model is loaded as a text generation pipeline using the default configuration described
    in the model card【554344751068985†L114-L141】.
    """
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map=device_map,
    )


if __name__ == "__main__":
    # Example usage for Gemma
    gemma_pipe = load_gemma()
    gemma_output = gemma_pipe("Hello, world!", max_new_tokens=50)
    print("Gemma output:", gemma_output)

    # Example usage for GPT-OSS
    gpt_oss_pipe = load_gpt_oss()
    messages = [
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ]
    gpt_oss_output = gpt_oss_pipe(messages, max_new_tokens=256)
    print("GPT-OSS output:", gpt_oss_output)
