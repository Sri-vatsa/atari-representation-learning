import torch 

from atari_rl.atariari.methods.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer

def train_encoder(encoder, tr_eps, val_eps, num_epochs, lr, patience, wandb, save_dir, batch_size, model_name, method="global-infonce-stdim"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observation_shape = tr_eps[0][0].shape
    config = {}
    config['epochs'] = num_epochs
    config['lr'] = lr
    config['patience'] = patience
    config['batch_size'] = batch_size
    config['save_dir'] = save_dir
    config['obs_space'] = observation_shape  # weird hack
    config['model_name'] = model_name

    # Add different training methods here
    if method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)

    # Implement save model
    trainer.train(tr_eps, val_eps)

    return encoder

