import openai


# Create a fine-tuning job
fine_tuning_response = openai.FineTune.create(
  training_file="file-abc123",  # Replace with your uploaded file ID
  model="davinci"  # Or another model if fine-tuning is available
)

print(fine_tuning_response)

response = openai.Completion.create(
  model="fine-tuned-model-id",  # Replace with your fine-tuned model ID
  prompt="Translate the following English text to French: 'Where is the nearest train station?'",
  max_tokens=60
)

print(response.choices[0].text.strip())
