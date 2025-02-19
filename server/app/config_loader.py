import os
from os import environ as env
import sys
import json
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForImage2Image,
    AltDiffusionImg2ImgPipeline,
    AmusedImg2ImgPipeline,
    FlaxStableDiffusionImg2ImgPipeline,
    FluxControlImg2ImgPipeline,
    FluxControlNetImg2ImgPipeline,
    FluxImg2ImgPipeline,
    IFImg2ImgPipeline,
    IFImg2ImgSuperResolutionPipeline,
    Kandinsky3Img2ImgPipeline,
    KandinskyImg2ImgCombinedPipeline,
    KandinskyImg2ImgPipeline,
    KandinskyV22ControlnetImg2ImgPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22Img2ImgPipeline,
    KolorsImg2ImgPipeline,
    LatentConsistencyModelImg2ImgPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    ShapEImg2ImgPipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3PAGImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionPAGImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPAGImg2ImgPipeline,
    StableDiffusionXLControlNetUnionImg2ImgPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    StableUnCLIPImg2ImgPipeline,
    StableDiffusionInpaintPipeline
)
from diffusers import DPMSolverMultistepScheduler,EulerAncestralDiscreteScheduler
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import os
from safetensors.torch import load_file as safe_load

# Use Hugging Face Diffusers pipelines
pipeline_mapping = {
    "AutoPipelineForImage2Image": AutoPipelineForImage2Image,
    "AltDiffusionImg2ImgPipeline": AltDiffusionImg2ImgPipeline,
    "AmusedImg2ImgPipeline": AmusedImg2ImgPipeline,
    "FlaxStableDiffusionImg2ImgPipeline": FlaxStableDiffusionImg2ImgPipeline,
    "FluxControlImg2ImgPipeline": FluxControlImg2ImgPipeline,
    "FluxControlNetImg2ImgPipeline": FluxControlNetImg2ImgPipeline,
    "FluxImg2ImgPipeline": FluxImg2ImgPipeline,
    "IFImg2ImgPipeline": IFImg2ImgPipeline,
    "IFImg2ImgSuperResolutionPipeline": IFImg2ImgSuperResolutionPipeline,
    "Kandinsky3Img2ImgPipeline": Kandinsky3Img2ImgPipeline,
    "KandinskyImg2ImgCombinedPipeline": KandinskyImg2ImgCombinedPipeline,
    "KandinskyImg2ImgPipeline": KandinskyImg2ImgPipeline,
    "KandinskyV22ControlnetImg2ImgPipeline": KandinskyV22ControlnetImg2ImgPipeline,
    "KandinskyV22Img2ImgCombinedPipeline": KandinskyV22Img2ImgCombinedPipeline,
    "KandinskyV22Img2ImgPipeline": KandinskyV22Img2ImgPipeline,
    "KolorsImg2ImgPipeline": KolorsImg2ImgPipeline,
    "LatentConsistencyModelImg2ImgPipeline": LatentConsistencyModelImg2ImgPipeline,
    "OnnxStableDiffusionImg2ImgPipeline": OnnxStableDiffusionImg2ImgPipeline,
    "ShapEImg2ImgPipeline": ShapEImg2ImgPipeline,
    "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
    "StableDiffusion3Img2ImgPipeline": StableDiffusion3Img2ImgPipeline,
    "StableDiffusion3PAGImg2ImgPipeline": StableDiffusion3PAGImg2ImgPipeline,
    "StableDiffusionControlNetImg2ImgPipeline": StableDiffusionControlNetImg2ImgPipeline,
    "StableDiffusionImg2ImgPipeline": StableDiffusionImg2ImgPipeline,
    "StableDiffusionPAGImg2ImgPipeline": StableDiffusionPAGImg2ImgPipeline,
    "StableDiffusionXLControlNetImg2ImgPipeline": StableDiffusionXLControlNetImg2ImgPipeline,
    "StableDiffusionXLControlNetPAGImg2ImgPipeline": StableDiffusionXLControlNetPAGImg2ImgPipeline,
    "StableDiffusionXLControlNetUnionImg2ImgPipeline": StableDiffusionXLControlNetUnionImg2ImgPipeline,
    "StableDiffusionXLImg2ImgPipeline": StableDiffusionXLImg2ImgPipeline,
    "StableDiffusionXLPAGImg2ImgPipeline": StableDiffusionXLPAGImg2ImgPipeline,
    "StableUnCLIPImg2ImgPipeline": StableUnCLIPImg2ImgPipeline
}

scheduler_mapping = {
    "DPM": DPMSolverMultistepScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler
}

os.environ["KERAS_BACKEND"] = "jax"

# ✅ Import Keras only if needed (to prevent unnecessary dependencies)
try:
    from keras_cv.models.stable_diffusion import StableDiffusion3ImageToImage
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# Set Hugging Face cache directory
HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
HF_CACHE = os.path.join(HF_HOME, "hub")

# Get API key for authentication
HF_TOKEN = os.getenv("HF_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("❌ Hugging Face API key not found. Set HF_API_KEY environment variable.")

# Load Config File
CONFIG_PATH = "server/config.json"


class ModelConfigLoader:
    def __init__(self, config_name: str):
        """ Loads the correct model configuration from config.json """
        
        try:
            with open(CONFIG_PATH, "r") as f:
                config_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ Error loading config file: {e}")

        self.config_name = config_name
        self.config_entry = config_data.get(self.config_name, {})
        if not self.config_entry:
            raise ValueError(f"❌ Model '{self.config_name}' not found in config.")

        self.model_name = self.config_entry["model_name"]
        self.final_model_name = self.config_entry["final_model_name"]
        self.final_model_checkpoint_file = f"{env.get('SAFETENSOR_HOME')}/{self.config_entry['final_model_checkpoint_file']}"
        
        self.pipeline_class = self.config_entry.get("pipeline_class", "StableDiffusionImg2ImgPipeline")
        self.final_pipeline_class = self.config_entry.get("final_pipeline_class", "StableDiffusionImg2ImgPipeline")
        self.pipeline = None
        
        # ✅ Enhanced GPU detection for H200 & other new architectures
        if torch.cuda.device_count() > 0:
            self.device = "cuda"
            print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)} (Using CUDA)")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Using Apple MPS")
        else:
            print("❌ ERROR: No compatible GPU detected. This program requires a CUDA or MPS-compatible device.")
            sys.exit(1)  # ✅ Terminate program immediately

    @staticmethod
    def apply_scheduler_to_pipeline(scheduler_config, pipeline):
        """
        Applies a scheduler to the given pipeline if the scheduler_config is provided.

        Parameters:
        - scheduler_config (dict or None): The scheduler configuration node from the config.
        - pipeline (DiffusionPipeline): The Stable Diffusion pipeline to modify.

        Returns:
        - None (modifies pipeline in-place).
        """
        global scheduler_mapping
        
        if not scheduler_config or not isinstance(scheduler_config, dict):
            print("⚠️ No scheduler configuration provided. Using default scheduler.")
            return

        scheduler_type = scheduler_config.get("type")
        scheduler_class = scheduler_mapping.get(scheduler_type)

        if not scheduler_class:
            print(f"⚠️ Warning: Unknown scheduler type '{scheduler_type}'. Skipping scheduler configuration.")
            return

        scheduler_kwargs = {k: v for k, v in scheduler_config.items() if k != "type"}

        try:
            # Use existing scheduler config if available
            pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **scheduler_kwargs)
            print(f"✅ Scheduler '{scheduler_type}' applied from model configuration.")
        except Exception:
            print(f"⚠️ Model does not have a scheduler_config.json. Initializing manually.")
            pipeline.scheduler = scheduler_class(**scheduler_kwargs)
          
    @staticmethod
    def get_torch_dtype_and_variant(config_entry):
        """
        Extracts the torch_dtype and variant from the configuration.

        Parameters:
        - config_entry (dict): The model configuration.

        Returns:
        - tuple (torch_dtype, variant): Processed values.
        """
        # Extract dtype from config
        dtype_str = config_entry.get("torch_dtype", "").strip().lower()
        torch_dtype = None if dtype_str == "" else torch.float16 if dtype_str == "float16" else torch.float32
        print(f"🔥 torch dtype: {torch_dtype}")

        # Extract variant from config
        variant = config_entry.get("variant", "").strip().lower()
        variant = None if variant == "" else variant
        print(f"🔥 variant: {variant}")

        return torch_dtype, variant


    @staticmethod
    def apply_lora_to_pipeline(lora_config, dtype, pipeline):
        """
        Applies a LoRA model to the given pipeline if the lora_config is provided.

        Parameters:
        - lora_config (dict or None): The LoRA configuration node from the config.
        - pipeline (DiffusionPipeline): The Stable Diffusion pipeline to modify.

        Returns:
        - None (modifies pipeline in-place).
        """
        if not lora_config:
            return  # No LoRA configured, leave pipeline as is.

        try:
            # Extract LoRA parameters
            lora_path = lora_config.get("model_path")
            alpha = lora_config.get("alpha", 1.0)  # Default scaling

            if not lora_path:
                print("⚠️ Warning: LoRA model path is missing in config.")
                return

            # Resolve the LoRA file path using SAFETENSOR_HOME
            safetensor_home = os.getenv("SAFETENSOR_HOME", "")
            possible_local_path = os.path.join(safetensor_home, lora_path)

            if os.path.isabs(possible_local_path) and os.path.exists(possible_local_path):
                lora_file = possible_local_path
                print(f"📂 Loading local LoRA from SAFETENSOR_HOME: {lora_file}")
            else:
                print(f"📥 Downloading LoRA from Hugging Face: {lora_path}")
                lora_file = hf_hub_download(repo_id=lora_path, filename=os.path.basename(lora_path))

            # Ensure the LoRA file exists before proceeding
            if not os.path.exists(lora_file):
                raise FileNotFoundError(f"❌ LoRA file not found: {lora_file}")

            # Verify LoRA file format
            use_safetensors = lora_file.endswith(".safetensors")

            # Determine appropriate torch dtype
            print(f"🔄 Loading LoRA with dtype={dtype}...")
            if dtype == "float16":
                torch_dtype = torch.float16
            elif dtype == "float32":
                torch_dtype = torch.float32
            else:
                raise ValueError(f"❌ Unsupported dtype: {dtype}. Use 'float16' or 'float32'.")

            pipeline.to(dtype=torch_dtype)

            # Apply LoRA to pipeline
            pipeline.load_lora_weights(lora_file, weight=alpha)
            print(f"✅ LoRA successfully loaded from {lora_file} with dtype={dtype} and alpha={alpha}")

        except Exception as e:
            print(f"❌ Error applying LoRA: {e}")
            
    def initialize_pipeline(self):
        global pipeline_mapping
        
        print("INPAINT PIPELINE")
        print(f"🚀 Loading model: {self.config_name} ({self.model_name})")
        print(f"📂 Cache directory: {HF_CACHE}")
        print(f"🔄 Using pipeline: {self.pipeline_class}")

        self.current_seed = self.config_entry.get("seed", None)

        try:
            # 🔹 Select appropriate pipeline
            if self.pipeline_class == "StableDiffusion3ImageToImage":
                if not KERAS_AVAILABLE:
                    raise ImportError("❌ Keras is not installed, but is required for StableDiffusion3ImageToImage.")

                print("🔄 Loading Keras StableDiffusion3ImageToImage model...")
                self.pipeline = StableDiffusion3ImageToImage.from_preset(self.model_name)

            else:
                PipelineClass = pipeline_mapping.get(self.pipeline_class)
                if not PipelineClass:
                    raise ValueError(f"❌ Invalid pipeline_class '{self.pipeline_class}' in config.")

                # 🔹 Get dtype and variant
                torch_dtype, variant = self.get_torch_dtype_and_variant(self.config_entry)
            
                # 🔹 Define scheduler mapping
                scheduler_mapping = {
                    "DPM": DPMSolverMultistepScheduler,
                    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
                }

                # 🔹 Load the pipeline
                self.pipeline = PipelineClass.from_pretrained(
                    self.model_name,
                    cache_dir=HF_CACHE,
                    token=HF_TOKEN,
                    torch_dtype=torch_dtype,  # Uses None if unspecified
                    variant=variant,  # Uses None if unspecified
                    use_safetensors=True,
                ).to(self.device)

                # 🔹 Apply scheduler (if configured)
                self.apply_scheduler_to_pipeline(self.config_entry.get("scheduler"), self.pipeline)
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model {self.config_name}: {e}")

        return self.pipeline

    def initialize_final_pipeline(self):
        global pipeline_mapping
        
        # 🔹 Select appropriate pipeline
        if self.final_pipeline_class == None:
            print("⚠️ No final pipeline class specified. Using default pipeline only.")
            return None
        
        print("FINAL PIPELINE")
        print(f"🚀 Loading model: {self.config_name} ({self.final_model_name})")
        print(f"📂 Cache directory: {HF_CACHE}")
        print(f"🔄 Using pipeline: {self.final_pipeline_class}")

        self.current_seed = self.config_entry.get("seed", None)

        try:
                
            PipelineClass = pipeline_mapping.get(self.final_pipeline_class)
            if not PipelineClass:
                raise ValueError(f"❌ Invalid pipeline_class '{self.pipeline_class}' in config.")

            # 🔹 Get dtype and variant
            torch_dtype, variant = self.get_torch_dtype_and_variant(self.config_entry)
        
            # 🔹 Define scheduler mapping
            scheduler_mapping = {
                "DPM": DPMSolverMultistepScheduler,
                "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            }

            # 🔹 Load the pipeline
            base_pipeline = PipelineClass.from_pretrained(
                self.final_model_name,
                cache_dir=HF_CACHE,
                token=HF_TOKEN,
                torch_dtype=torch_dtype,  # Uses None if unspecified
                variant=variant,  # Uses None if unspecified
                use_safetensors=True,
            ).to(self.device)

            # 🔹 Apply scheduler (if configured)
            self.apply_scheduler_to_pipeline(self.config_entry.get("scheduler"), self.pipeline)
        
            if self.final_model_checkpoint_file is not None:
                print(f"🔹 Loading final model checkpoint from {self.final_model_checkpoint_file}")
                # Then, load the checkpoint from your safetensors file.
                checkpoint = safe_load(self.final_model_checkpoint_file)

                # Now, apply the checkpoint to the appropriate part of the model.
                # For example, if your checkpoint contains weights for the UNet:
                base_pipeline.unet.load_state_dict(checkpoint, strict=False)
            else:
                print("⚠️ No final model checkpoint file specified. Using base pipeline only.")
            
            # Apply LoRA (if configured)
            self.apply_lora_to_pipeline(self.config_entry.get("lora"), self.config_entry.get("torch_dtype"), base_pipeline)
           
            self.final_pipeline = base_pipeline
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model {self.config_name}: {e}")

        return self.final_pipeline
    
    def get_parameters(self):
        """ Retrieves model-specific generation parameters from config. """
        return self.config_entry.get("parameters", {})
    
    def get_bg_parameters(self):
        """ Retrieves model-specific background generation parameters from config. """
        return self.config_entry.get("bg_parameters", {})
    
    def get_final_parameters(self):
        """ Retrieves final composite parameters """
        return self.config_entry.get("final_parameters", {})