from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "/ytech_m2v5_hdd/workspace/kling_mm/Models/Kimi-VL-A3B-Thinking-2506"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image_paths = ["/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/diffusers/examples/text_to_image/generated_images_origin_1024/a_beautiful_landscape_002.png"]
images = [Image.open(path) for path in image_paths]
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path} for image_path in image_paths
        ] + [{"type": "text", "text": "Please infer step by step who this manuscript belongs to and what it records"}],
    },
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
inputs = processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
for i in inputs["input_ids"][0]:
    print(i)

generated_ids = model.generate(**inputs, max_new_tokens=32768, temperature=0.8)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)
