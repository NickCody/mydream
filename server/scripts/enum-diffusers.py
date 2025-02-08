import diffusers

# List all available attributes in diffusers
available_pipelines = [attr for attr in dir(diffusers) if "Pipeline" in attr]

print("Available pipelines in your installed version of diffusers:")
for pipeline in available_pipelines:
    print(f"  - {pipeline}")
    
