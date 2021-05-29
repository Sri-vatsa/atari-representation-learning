import torch 

from atari_rl.atariari.methods.global_infonce_stdim import CLIPGlobalInfoNCESpatioTemporalTrainer
from atari_rl.atariari.methods.global_local_infonce import CLIPGlobalLocalInfoNCESpatioTemporalTrainer
from atari_rl.atariari.methods.cpc_clip import CLIPCPCTrainer

def run_encoder_training(encoder, tr_eps, val_eps, config, wandb, method="global-infonce-stdim"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add different training methods here
    if method == "global-infonce-stdim":
        trainer = CLIPGlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif method == 'clip-cpc':
        trainer = CLIPCPCTrainer(encoder, config, device=device, wandb=wandb)
    elif method == "stdim":
        trainer = CLIPInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    else:
        raise Exception("Invalid method...please pick a valid encoder training method")

    # Implement save model
    trainer.train(tr_eps, val_eps)

    return encoder
