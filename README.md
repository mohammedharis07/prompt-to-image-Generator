# prompt-to-image-Generator
A powerful AI-based image generator using Stable Diffusion. Convert text prompts into high-quality images with support for multiple artistic styles, including Realistic, Anime, Fantasy, and Pixel Art. Optimized for speed with GPU acceleration, FP16 precision, and xFormers support. Customizable settings for fine-tuned image generation.

🚀 Features

Multiple Styles: Realistic, Anime, Fantasy, Pixel Art.

Optimized for Speed: Uses GPU (CUDA), FP16, and xFormers for faster generation.

Customizable Settings: Adjust inference steps, guidance scale, and resolution.

Negative Prompts Support: Helps remove unwanted elements from images.

Automatic GPU Selection: Runs on CUDA if available, otherwise uses CPU.

📌 Installation

1️⃣ Install Dependencies

Make sure you have Python installed (preferably Python 3.9+).
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate xformers
2️⃣ Install CUDA (For NVIDIA GPU users)

Check if CUDA is installed:

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))

If False, install CUDA 12.1.

⚡ Optimizations for Speed

Reduce Inference Steps: Set num_inference_steps=20-30 (Default is 50, which is slower).

Use FP16 Precision: torch_dtype=torch.float16 speeds up GPU processing.

Enable xFormers: pipe.enable_xformers_memory_efficient_attention() reduces memory usage.

Lower Image Resolution: Use height=384, width=384 for faster generation.

🛠 Troubleshooting

1️⃣ CUDA Not Available?

Run python -c "import torch; print(torch.cuda.is_available())".

If False, install CUDA 12.1 and reinstall PyTorch with CUDA support.

2️⃣ Image Generation Too Slow?

Lower num_inference_steps.

Reduce image resolution.

Ensure xFormers is installed: pip install xformers.

3️⃣ Out of Memory (OOM) Error?

Use torch_dtype=torch.float16 to reduce VRAM usage.

Restart your system to free up memory.

📜 License

This project is open-source and available under the MIT License.

🛠 Running the Code from GitHub

Anyone can run this code by following these steps:

Clone the Repository

git clone https://github.com/your-username/prompt-to-image.git
cd prompt-to-image

Install Dependencies

pip install -r requirements.txt

Run the Script

python generate.py



🌟 Contributing

Feel free to open an issue or submit a pull request if you want to improve the project!

Drawback

the only issue is that of the image generation taking more time as we are opting for Stable Diffusion model

