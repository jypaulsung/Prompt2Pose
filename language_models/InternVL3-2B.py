import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import gradio as gr

# Image Preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def preprocess_single_image(pil_img, input_size=448):
    """
    Process a single image for the model.
    """
    transform = build_transform(input_size)
    pil_img = pil_img.resize((input_size, input_size))
    tensor_img = transform(pil_img).unsqueeze(0)  # shape: (1, 3, H, W)
    return tensor_img

# Loading the model
model_path = "OpenGVLab/InternVL3-2B"
# model_path = "OpenGVLab/InternVL3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    # load_in_8bit=True, # 8-bit quantization for 8B model
    low_cpu_mem_usage=True,
    use_flash_attn=torch.cuda.is_available(),
    trust_remote_code=True
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False
)
print("Model loaded successfully!")

# Conversation Helpers
def pairs_to_messages(history_pairs):
    """
    Convert a list of (user, assistant) pairs into a list of messages.
    """
    messages = []
    for user_msg, assistant_msg in history_pairs:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    return messages

def chat_interface(query, image, history_pairs):
    """
    Main function called on each user submit.
    """
    generation_config = {"max_new_tokens": 1024, "do_sample": True}

    # Preprocess image if provided
    pixel_values = None
    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Make sure the image is handled correctly
        try:
            pixel_values = preprocess_single_image(image, 448)
            pixel_values = pixel_values.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device)
            # Add <image> tag to the query if it's not already there
            if "<image>" not in query:
                query = "<image>\n" + query
        except Exception as e:
            print(f"Error processing image: {e}")
            # Continue without the image if there's an error
            pixel_values = None
    
    # Model chat call
    try:
        response, updated_history = model.chat(
            tokenizer,
            pixel_values,
            query,
            generation_config,
            history=history_pairs,       # pass the old history pairs
            return_history=True          # get the entire updated history as (user, assistant) pairs
        )
    except Exception as e:
        print(f"Error during model.chat: {e}")
        # Provide a fallback response if there's an error
        if history_pairs is None:
            history_pairs = []
        updated_history = history_pairs + [(query, "I'm sorry, I encountered an error processing your request.")]
        response = "I'm sorry, I encountered an error processing your request."
    
    # Return updated conversation in two forms:
    # 1) messages for the Chatbot's display
    # 2) the raw pairs for state tracking
    return pairs_to_messages(updated_history), updated_history

def reset_history():
    """
    Returns empty conversation for messages + state.
    """
    return [], []

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# InternVL3 Chat Interface")
    
    # 'type="messages"' => expects a list of dicts with 'role'/'content'
    chatbot = gr.Chatbot(label="Conversation", type="messages")
    # We'll store conversation internally as a list of (user, assistant) pairs
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(placeholder="Enter your message here", label="Message")
        # Single image only (no `multiple=True`) to avoid errors
        img_input = gr.Image(label="Optional Image Input", type="pil")

    with gr.Row():
        send_btn = gr.Button("Send")
        reset_btn = gr.Button("Reset Conversation")

    # On click, process chat -> updates the chatbot display and conversation state
    send_btn.click(
        fn=chat_interface,
        inputs=[txt, img_input, state],
        outputs=[chatbot, state],
        scroll_to_output=True
    )

    # Reset the chat conversation
    reset_btn.click(fn=reset_history, outputs=[chatbot, state])

if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch()