'''
This script is for testing the Qwen2.5-Math-7B-Instruct model
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def run_inference():
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Horizontal line with a y-coordinate that equals the average of the original y-coordinates
    prompt = (
        "You are given n-dots located at the following pixel coordinates: "
        "(342, 338), (224, 300), and (242, 172). "
        "Rearrange these dots to satisfy the following conditions: "
        "1. Each new dot should be within the image boundary of (0, 0) to (512, 512). "
        "2. All dots should have the same y-coordinate which is the average of the original y-coordinates. "
        "3. The average of the n-dots' new x-coordinates should be 256. "
        "4. Each dot should be 50 pixels apart from its neighboring dot. "
        "5. Assign new coordinates to the originals dots one by one. Among the n-possible options, it should be assigned to the one that is the closest to the new coordinate. "
        "6. Round the new coordinates to the nearest integer. "
        "Provide the new coordinates for each dot in the order of the original coordinates in the following format: "
        "Original coordinate: (x1, y1) -> New coordinate: (x2, y2)"
    )

    # CoT
    messages = [
        {"role": "system", "content": "Please reason step by step."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    del model
    del tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_inference()