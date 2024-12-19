# from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
# from diffusers import StableDiffusion3Pipeline
# import torch

# model_id = "stabilityai/stable-diffusion-3.5-large"

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# model_nf4 = SD3Transformer2DModel.from_pretrained(
#     model_id,
#     subfolder="transformer",
#     quantization_config=nf4_config,
#     torch_dtype=torch.bfloat16
# )

# pipeline = StableDiffusion3Pipeline.from_pretrained(
#     model_id, 
#     transformer=model_nf4,
#     torch_dtype=torch.bfloat16
# )
# pipeline.enable_model_cpu_offload()

# prompt = "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus, basking in a river of melted butter amidst a breakfast-themed landscape. It features the distinctive, bulky body shape of a hippo. However, instead of the usual grey skin, the creature's body resembles a golden-brown, crispy waffle fresh off the griddle. The skin is textured with the familiar grid pattern of a waffle, each square filled with a glistening sheen of syrup. The environment combines the natural habitat of a hippo with elements of a breakfast table setting, a river of warm, melted butter, with oversized utensils or plates peeking out from the lush, pancake-like foliage in the background, a towering pepper mill standing in for a tree.  As the sun rises in this fantastical world, it casts a warm, buttery glow over the scene. The creature, content in its butter river, lets out a yawn. Nearby, a flock of birds take flight"

# image = pipeline(
#     prompt=prompt,
#     num_inference_steps=28,
#     guidance_scale=4.5,
#     max_sequence_length=512,
# ).images[0]
# image.save("whimsical.png")

import torch
torch.cuda.empty_cache() 

from accelerate import infer_auto_device_map, init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
from utils.load_save_utils import *
import torch
from diffusers import FluxPipeline

# checkpoint = "black-forest-labs/FLUX.1-dev"
# pipe = FluxPipeline.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,device_map = 'balanced',offload_buffers=True)
# pipe.to('cuda:2')
# pipe.text_encoder_2.to('cuda:2')
# device_map = infer_auto_device_map(pipe.transformer)
# weights_location = hf_hub_download(checkpoint, filename="transformer/diffusion_pytorch_model-00001-of-00003.safetensors")

# device_map = infer_auto_device_map(pipe.transformer)

# # Remove GPU 4 (index 3) from the device map by reassigning layers to other GPUs
# # Here we create a new map and move layers that would be assigned to GPU 4
# # for layer_name, device in device_map.items():
# #     if device == 1:  # GPU 4
# #         # Assign the layer to the next available GPU (GPU 0, 1, or 2)
# #         new_device = (device + 1) % 3  # Wrap around and use GPU 0, 1, or 2
# #         device_map[layer_name] = new_device
# pipe.text_encoder.to('cpu')
# pipe.text_encoder_2.to('cpu')

# pipe.transformer = load_checkpoint_and_dispatch(
#     pipe.transformer,
#     weights_location,
#     device_map=device_map,
#     offload_folder="offload",
#     offload_buffers=True
# )
# device_map = infer_auto_device_map(pipe.transformer)
# weights_location = hf_hub_download(checkpoint, filename="text_encoder_2/model-00001-of-00002.safetensors")

# # pipe.vae.to('cuda:2')
def visualize(maps,
                      save_dir,
                      file_name,):


        # maps = maps.astype(np.uint16)
        # Plot the tensor as a heatmap
        plt.imshow(maps, cmap='viridis', interpolation='none')
        plt.colorbar()

        os.makedirs(save_dir,exist_ok=True)
        plt.savefig(f'{save_dir}/{file_name}.png')
        plt.close()

generate_im = True

if generate_im:
    pipe = load_model(model_name='flux-dev',device='balanced')
    pipe.attn_fetch_x.set_processor(transformer = pipe.transformer, processor_name = 'processor_flux_x')
    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=20,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save("flux-dev.png")


    maps =pipe.attn_fetch_x.maps_by_block() 


    save_pkl(data=maps, directory = 'generation_outputs/maps',file_name = 'test')

results = False 
if results:
    
    data = load_pkl(directory='generation_outputs/maps', file_name='test')
    
    for i in range(20):
        arr = data['block_9'][i]
        # scale = np.mean(np.array([arr[i,i] for i in range(4608)]))
        # arr -= scale * np.eye(4608)
        # breakpoint()
        img = arr[1312,512:].reshape(64,64)
        # arr = arr[8]
        # plt.hist(arr.flatten(), bins = 100)
        
        # plt.yscale('log')
        # plt.yscale('log')
        # plt.savefig(f'generation_outputs/visualizations/histtime_{i}.png')
        # plt.close()
    # print(arr)
    # exit()
        visualize(maps = img,save_dir = 'generation_outputs/visualizations',file_name=f'time_{i}')