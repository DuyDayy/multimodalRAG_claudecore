import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import pandas as pd
import time

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load CLIP (Standard)
    print("Loading Standard CLIP (openai/clip-vit-base-patch32)...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()

    # 2. Load SigLIP (Your Model)
    print("Loading SigLIP (google/siglip-so400m-patch14-384)...")
    siglip_model_name = "google/siglip-so400m-patch14-384"
    siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
    siglip_model = AutoModel.from_pretrained(siglip_model_name).to(device)
    siglip_model.eval()

    # Create a synthetic image
    img_array = np.ones((384, 384, 3), dtype=np.uint8) * 200 # light gray background
    img_array[250:350, 250:350, :] = [255, 0, 0] # red square
    image = Image.fromarray(img_array)
    
    text_queries = [
        "a red square", 
        "a light gray background",
        "a blue circle" # negative example
    ]

    print("Running Standard CLIP...")
    t0 = time.time()
    clip_inputs = clip_processor(text=text_queries, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
        image_embeds = clip_outputs.image_embeds / clip_outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = clip_outputs.text_embeds / clip_outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        clip_sims = torch.matmul(image_embeds, text_embeds.t())[0].cpu().numpy()
    clip_time = time.time() - t0

    print("Running Your Model (SigLIP + 6 Crops MaxSim)...")
    t0 = time.time()
    w, h = image.size
    cw, ch = w // 2, h // 2
    crops = [
        image, 
        image.crop((0, 0, cw, ch)), 
        image.crop((cw, 0, w, ch)), 
        image.crop((0, ch, cw, h)), 
        image.crop((cw, ch, w, h)), # This crop contains the red square perfectly
        image.crop((cw//2, ch//2, w - cw//2, h - ch//2))
    ]
    
    siglip_text_inputs = siglip_processor(text=text_queries, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        text_outputs = siglip_model.get_text_features(**siglip_text_inputs)
        text_features = text_outputs if isinstance(text_outputs, torch.Tensor) else text_outputs.text_embeds if hasattr(text_outputs, 'text_embeds') else text_outputs.pooler_output
        siglip_text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        siglip_img_inputs = siglip_processor(images=crops, return_tensors="pt").to(device)
        img_outputs = siglip_model.get_image_features(**siglip_img_inputs)
        img_features = img_outputs if isinstance(img_outputs, torch.Tensor) else img_outputs.image_embeds if hasattr(img_outputs, 'image_embeds') else img_outputs.pooler_output
        siglip_img_embeds = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        
        sim_matrix = torch.matmul(siglip_img_embeds, siglip_text_embeds.t())
        your_sims = sim_matrix.max(dim=0).values.cpu().numpy()
    your_time = time.time() - t0

    results = []
    for i, q in enumerate(text_queries):
        results.append({
            "Query": f'"{q}"',
            "Standard CLIP Cosine": f"{clip_sims[i]:.4f}",
            "Your Model (MaxSim)": f"{your_sims[i]:.4f}",
            "Delta": f"{your_sims[i] - clip_sims[i]:+.4f}"
        })

    # Manual Markdown Table
    md_table = "| Query | Standard CLIP Cosine | Your Model (MaxSim) | Delta |\n"
    md_table += "|---|---|---|---|\n"
    for r in results:
        md_table += f"| {r['Query']} | {r['Standard CLIP Cosine']} | {r['Your Model (MaxSim)']} | {r['Delta']} |\n"


    
    out_str = "### Bảng kết quả chạy thử nghiệm\n\n"
    out_str += f"**Mô phỏng ảnh:** Nền xám (background lớn) chứa 1 hình vuông đỏ (vật thể mục tiêu) ở góc dưới phải.\n\n"
    out_str += md_table + "\n\n"
    out_str += f"- **Thời gian chạy CLIP:** {clip_time:.4f}s\n"
    out_str += f"- **Thời gian chạy Your Model (6 crops):** {your_time:.4f}s\n"
    
    with open("comparison_results.md", "w") as f:
        f.write(out_str)
        
    print("Done. Results written to comparison_results.md")

if __name__ == "__main__":
    main()
