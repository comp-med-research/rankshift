"""
Wrapper around OmniDocBench's GOT_img2md.py
Run with: .venv-got  (needs GOT-OCR2.0 repo on sys.path)
"""
import argparse, sys
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--model-dir",  type=Path, required=True,
                    help="Path to ucaslcl/GOT-OCR2_0 snapshot")
    ap.add_argument("--got-dir",    type=Path,
                    default=Path.home() / "projects" / "GOT-OCR2",
                    help="GOT-OCR2.0 repo root (for GOT package)")
    args = ap.parse_args()

    sys.path.insert(0, str(args.got_dir))

    import torch
    from transformers import AutoTokenizer
    from GOT.model import GOTQwenForCausalLM
    from GOT.utils.utils import disable_torch_init, KeywordsStoppingCriteria
    from GOT.utils.conversation import conv_templates, SeparatorStyle
    from GOT.model.plug.blip_process import BlipImageEvalProcessor
    from PIL import Image

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(
        str(args.model_dir), low_cpu_mem_usage=True,
        device_map="cuda", use_safetensors=True, pad_token_id=151643).eval()
    model.to(device="cuda", dtype=torch.bfloat16)

    image_processor      = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
    DEFAULT_IM_START_TOKEN    = "<img>"
    DEFAULT_IM_END_TOKEN      = "</img>"
    image_token_len = 256

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[got] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="got", unit="img"):
        try:
            image = Image.open(img_path).convert("RGB")

            qs = ("OCR with format: " +
                  DEFAULT_IM_START_TOKEN +
                  DEFAULT_IMAGE_PATCH_TOKEN * image_token_len +
                  DEFAULT_IM_END_TOKEN + "\n" + "OCR with format: ")

            conv = conv_templates["mpt"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = tokenizer([prompt])
            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            image_tensor      = image_processor(image)
            image_tensor_high = image_processor_high(image.copy())

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    images=[(image_tensor.unsqueeze(0).half().cuda(),
                             image_tensor_high.unsqueeze(0).half().cuda())],
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=20,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping])

            output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            (args.out_dir / f"{img_path.stem}.md").write_text(output, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
