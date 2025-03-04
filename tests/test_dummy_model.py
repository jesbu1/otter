import torch
import clip
import tyro
from otter.policy.otter import OTTER
from otter.util.args import ExperimentConfig

def test_otter(args: ExperimentConfig):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = OTTER(
        model_config=args.model_cfg,
        shared_config=args.shared_cfg,
    ).to(device)
    print(model)
    # print the number of trainable parameters vs total number of parameters 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # print percentage of trainable 
    print(f"Trainable parameter percentage: {trainable_params/total_params*100:.2f}%")
    
    # Create dummy inputs
    batch_size = 2
    context_length = 12
    image_size = args.shared_cfg.image_size  # CLIP ViT-L/14 input size
    
    images = torch.randn(batch_size, context_length, 2, 3, image_size, image_size)
    # images to 0, 255 range and move to device 
    images = (images + 1) / 2 * 255
    images = images.to(torch.uint8).to(device)
    
    # Create dummy text tokens using CLIP's tokenizer
    text = clip.tokenize(["pick up the blue cube", "place the red ball"]).to(device)
    
    # 10-dim proprioception vector
    proprio = torch.randn(batch_size, context_length, args.model_cfg.proprio_input_dim).to(device)  
    
    # gt action shape: (batch_size, context_length, action_horizon, action_dim)
    gt_actions = torch.randn(batch_size, context_length, args.shared_cfg.action_horizon, args.model_cfg.action_dim).to(device)
    
    # Test forward pass
    try:
        loss = model.forward(images, text, proprio, gt_actions)
        loss.backward()
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e
    
    # Test gradient flow
    # Check gradients of a few key components
    components_to_check = [
        ('temperature', model.visual_extraction[0].temperature),
        ('vision_pooling', model.vision_poolings[0].query),
        ('text_pooling', model.text_pooling.query),
        ('action_head', list(model.action_head.parameters())[0])
    ]
    
    print("\nGradient check:")
    for name, param in components_to_check:
        if param.grad is not None:
            print(f"{name} gradient norm: {param.grad.norm().item()}")
            assert param.grad.norm().item() > 0, f"No gradient flow through {name}"
        else:
            print(f"Warning: {name} has no gradient")
            
if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    test_otter(args)