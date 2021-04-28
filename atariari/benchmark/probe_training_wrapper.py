from .probe import ProbeTrainer

# train using embeddings
def train_embeddings(encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size, 
                 tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels, 
                 tr_eps_tensors, val_eps_tensors, tr_lbls, val_lbls, test_eps_tensors, test_lbls
                 ):
    probe_trainer = ProbeTrainer(encoder=None,
                          epochs=num_epochs,
                          lr=lr,
                          batch_size=batch_size,
                          patience=patience,
                          wandb=wandb,
                          fully_supervised=False,
                          save_dir=save_dir,
                          representation_len=encoder.feature_size)
    probe_trainer.train(tr_episodes, val_episodes,
                      tr_labels, val_labels, batched_tr_emb=tr_eps_tensors, batched_val_emb=val_eps_tensors, batched_tr_labels=tr_lbls, batched_val_labels=val_lbls)

    final_accuracies, final_f1_scores = probe_trainer.test(test_episodes, test_labels, batched_emb=test_eps_tensors, batched_labels=test_lbls)

    wandb.log(final_accuracies)
    wandb.log(final_f1_scores)

# train using images
def train_images(encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size,
                 tr_episodes, val_episodes, tr_labels, val_labels, 
                 test_episodes, test_labels):
  
    probe_trainer = ProbeTrainer(encoder=encoder,
                          epochs=num_epochs,
                          lr=lr,
                          batch_size=batch_size,
                          patience=patience,
                          wandb=wandb,
                          fully_supervised=False,
                          save_dir=save_dir,
                          representation_len=encoder.feature_size)
    probe_trainer.train(tr_episodes, val_episodes,
                      tr_labels, val_labels)

    final_accuracies, final_f1_scores = probe_trainer.test(test_episodes, test_labels)

    wandb.log(final_accuracies)
    wandb.log(final_f1_scores)

# main training method
def run_training(training_input, encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size, 
                 tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels, 
                 tr_eps_tensors, val_eps_tensors, tr_lbls, val_lbls, test_eps_tensors, test_lbls):
  
  if training_input == 'embeddings':
    train_embeddings(encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size,
                 tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels, 
                 tr_eps_tensors, val_eps_tensors, tr_lbls, val_lbls, test_eps_tensors, test_lbls)
  elif training_input == 'images':
    train_images(encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size,
                 tr_episodes, val_episodes, tr_labels, val_labels, 
                 test_episodes, test_labels)
  else:
    print("Invalid input...choose either 'embeddings' and 'images'")