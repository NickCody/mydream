def generate_prompt(transcription: str, default_prompt: str = "a portrait with creative lighting") -> str:
    """
    Converts raw transcription text into a creative prompt.
    
    Args:
        transcription (str): The raw text captured from speech recognition.
        default_prompt (str): The fallback prompt if no transcription is provided.
        
    Returns:
        str: A creative prompt derived from the transcription.
    """
    transcription = transcription.strip()
    if transcription:
        # Here, you might add more advanced NLP or template logic.
        # For simplicity, we return a prompt that integrates the transcription.
        return f"a person with {transcription}"
    return default_prompt


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Test the prompt generator with sample input
    sample_text = "vivid red hair and striking green eyes"
    prompt = generate_prompt(sample_text)
    print("Generated Prompt:", prompt)