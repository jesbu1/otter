import os 
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import tyro
    
def crop2square(pil_img):
    W, H = pil_img.size
    if H == W:
        return pil_img
    elif H > W:
        return pil_img.crop((0, (H - W) // 2, W, W + (H - W) // 2))
    else:
        return pil_img.crop(((W - H) // 2, 0, H + (W - H) // 2, H))

def main(
    text: str, # text prompt
    image: str, # path to image
    out_dir: str = "visualization", # output directory
    model_name: str = "ViT-L/14" # model name
):

    os.makedirs(out_dir, exist_ok=True)
    print("Output directory: ", out_dir)
    image = Image.open(image)
    image = crop2square(image)

    # make output dir
    os.makedirs(out_dir, exist_ok=True)

    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    print(preprocess)

    # Load and preprocess the image
    first_k_tokens = 15
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Preprocess the text
    # create a hook to extract image patches 
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output
        return hook
    text_input = clip.tokenize([text]).to(device)

    model.transformer.register_forward_hook(getActivation('text_features'))
    # Encode the text
    with torch.no_grad():
        text_out_feat = model.encode_text(text_input) # (1, 768)
        text_out_feat = text_out_feat / text_out_feat.norm(dim=-1, keepdim=True)

    # text features 
    text_features = activation['text_features'].permute(1, 0, 2)[:, :first_k_tokens] 
    text_features = model.ln_final(text_features).type(model.dtype) @ model.text_projection
    text_features = text_features.squeeze()

    # normalize_text_features (77, 512)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # register the hook
    model.visual.transformer.register_forward_hook(getActivation('image_patches_out'))
    model.visual.transformer.resblocks[-1].attn.register_forward_hook(getActivation('image_patches'))
    with torch.no_grad():
        image_patches = model.encode_image(image_input)

    patch_features_out = activation["image_patches_out"]
    patch_features = activation["image_patches"][0]

    # extract patch features 
    patch_features = patch_features.permute(1, 0, 2) 
    patch_features = model.visual.ln_post(patch_features)

    patch_features_out = patch_features_out.permute(1, 0, 2) 
    patch_features_out = model.visual.ln_post(patch_features_out)

    if model.visual.proj is not None:
        patch_features = patch_features @ model.visual.proj
        patch_features_out = patch_features_out @ model.visual.proj

    patch_features = patch_features.squeeze()
    patch_features_out = patch_features_out.squeeze()

    # normalize patch features
    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
    patch_features_out = patch_features_out / patch_features_out.norm(dim=-1, keepdim=True)

    # separate cls token 
    cls_token = patch_features[:1] # (1, 512)
    image_patch_features = patch_features[1:] # (196, 512)
    patch_features_out = patch_features_out[1:] # (196, 512)

    # for each text token, visualize the similarity with each image patch (cosine similarity)
    # text_features = torch.concat([text_features, text_out_feat], dim=0)
    similarity = text_features @ image_patch_features.T # CLEARCLIP
    # similarity = text_features @ patch_features_out.T # CLIP 
    similarity_global = text_out_feat @ image_patch_features.T
    similarity = torch.cat([similarity, similarity_global], dim=0)
    # save text_features and image_patch_features as numpys 
    np.save(f'{out_dir}/text_features.npy', text_features.detach().cpu().numpy())
    np.save(f'{out_dir}/image_patch_features.npy', image_patch_features.detach().cpu().numpy())

    # temperature = 0.07
    temperature = 0.005
    similarity /= temperature
    similarity = similarity.softmax(dim=-1)

    # show a horizontal plot for the similarity. In the x axis, there should be 77 ticks. Each tick should represent a text token. Label it with the decoded text token
    # Decode the text tokens
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    tokens = []
    for i in range(len(text_input[0])):
        if i == first_k_tokens:
            break
        tokens.append(tokenizer.decode(text_input[0][i:i+1].cpu().numpy()))
    tokens.append("Text CLS Token")

    image = np.array(image)
    if "336" in model_name:
        featmap_wh = 24
    elif "ViT-B" in model_name:
        featmap_wh = 14
    else:
        featmap_wh = 16

    for i in trange(similarity.shape[0]):
        # show 3 plots: overlay, image, and similarity and colorbar
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Overlay plot
        ax[0].imshow(image)
        
        # Resize the heatmap to match the image dimensions
        heatmap = similarity[i].reshape(featmap_wh, featmap_wh).detach().cpu().numpy()
        heatmap_resized = np.repeat(np.repeat(heatmap, image.shape[0]//featmap_wh, axis=0), image.shape[1]//featmap_wh, axis=1)
        
        # # Create a mask for values below a certain threshold (e.g., 0.5)
        # mask = heatmap_resized < 0.5
        
        # # Apply the mask to the heatmap
        # masked_heatmap = np.ma.array(heatmap_resized, mask=mask)
        masked_heatmap = heatmap_resized
        
        # Plot the masked heatmap
        im = ax[0].imshow(masked_heatmap, alpha=0.6, cmap='viridis')
        ax[0].set_title(f"Token: {tokens[i]}")
        ax[0].axis('off')
        
        # Image plot
        ax[1].imshow(image)
        ax[1].set_title("Original Image")
        ax[1].axis('off')
        
        # Similarity plot
        im2 = ax[2].imshow(heatmap, cmap='viridis')
        ax[2].set_title("Similarity Heatmap")
        ax[2].axis('off')
        
        # Add colorbar
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        
        plt.savefig(f'{out_dir}/patch_similarity_{i}.png')


    # for each text token, visualize the similarity with the cls token
    cls_similarity = text_features @ cls_token.T
    cls_similarity = cls_similarity.T

    # Plot the similarity
    plt.figure(figsize=(15, 5))
    plt.imshow(cls_similarity.detach().cpu().numpy())
    plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=90)
    plt.xlabel("Text Tokens")
    plt.ylabel("Similarity")
    plt.colorbar()
    plt.title("Similarity between Image Patches and Text Tokens")
    plt.savefig(f'{out_dir}/text_similarity.png')

if __name__ == "__main__":
    tyro.cli(main)