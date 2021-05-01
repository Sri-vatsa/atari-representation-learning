import numpy as np
import os
import torch

from .episodes import get_episodes

def save_npy(filepath, data):
	np.savez_compressed(filepath, data=data)
	print("Data saved to {}".format(filepath))

def load_npy(filepath, file_name='arr_0'):
	loaded_data = np.load(filepath, allow_pickle=True, mmap_mode="r")
	print("Data loaded from {}".format(filepath))
	return loaded_data[file_name]

def get_episode_data(images_n_labels_dir, env_name, steps, collect_mode, color=True):
  try:
    tr_episodes = load_npy(os.path.join(images_n_labels_dir, "train_eps.npz"))
    tr_labels = load_npy(os.path.join(images_n_labels_dir, "train_labels.npz"))
    val_episodes = load_npy(os.path.join(images_n_labels_dir, "val_eps.npz"))
    val_labels = load_npy(os.path.join(images_n_labels_dir, "val_labels.npz"))
    test_episodes = load_npy(os.path.join(images_n_labels_dir, "test_eps.npz"))
    test_labels =load_npy(os.path.join(images_n_labels_dir, "test_labels.npz"))
  except:
    print("Unable to load data from drive...")
    tr_episodes, val_episodes,\
    tr_labels, val_labels,\
    test_episodes, test_labels = get_episodes(env_name=env_name, 
                                        steps=steps, 
                                        collect_mode=collect_mode,
                                        color=color)
  return tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels

def get_embedding_data(embeddings_dir):
  try:
    tr_episodes = torch.load(os.path.join(embeddings_dir, "clip_embeddings_train"))
    tr_labels = load_npy(os.path.join(embeddings_dir, "train_labels.npz"))
    val_episodes = torch.load(os.path.join(embeddings_dir, "clip_embeddings_val"))
    val_labels = load_npy(os.path.join(embeddings_dir, "val_labels.npz"))
    test_episodes = torch.load(os.path.join(embeddings_dir, "clip_embeddings_test"))
    test_labels = load_npy(os.path.join(embeddings_dir, "test_labels.npz"))

  except:
    raise Exception("Unable to load embedding data from drive...")

  return tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels

def get_data(data_type, data_dir, env_name, steps, collect_mode, color=True):
  if data_type == "embeddings":
    tr_episodes, val_episodes,\
    tr_labels, val_labels,\
    test_episodes, test_labels = get_embedding_data(data_dir)
  elif data_type == "images":
    tr_episodes, val_episodes,\
    tr_labels, val_labels,\
    test_episodes, test_labels = get_episode_data(data_dir, env_name=env_name, steps=steps, collect_mode=collect_mode, color=True)
  else:
    raise Exception("Invalid data type... choose between 'embeddings' & 'images'")
  
  return tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels

'''
def np_to_tensor(tr_eps, val_eps, test_eps):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  tr_eps_tensors = [torch.from_numpy(np.array(x)).to(device) for x in tr_eps]
  val_eps_tensors = [torch.from_numpy(np.array(x)).to(device) for x in val_eps]
  test_eps_tensors = [torch.from_numpy(np.array(x)).to(device) for x in test_eps]
  return tr_eps_tensors, val_eps_tensors, test_eps_tensors
'''