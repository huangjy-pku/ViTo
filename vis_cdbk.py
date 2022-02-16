"""
This script is used to visualize codebook learned by VQGAN, to verify whether it is in proper distribution
Show statistical results in three ways: code frequency histogram, distribution of vector norm, t-SNE visualization of embedding space
"""
import numpy as np
import torch
import os
from sklearn.manifold import TSNE
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt


def index_freq(seq_dir, codebook_size) -> np.ndarray:
    freq_list = np.zeros(codebook_size)
    print(f'Counting frequency in directory {seq_dir}')
    for fname in tqdm(os.listdir(seq_dir)):
        meta_seq = torch.load(os.path.join(seq_dir, fname), map_location='cpu')['answer']
        meta_seq = word_tokenize(meta_seq)
        for token in meta_seq:
            if 'code_' in token:
                idx = int(token[5:])
                freq_list[idx] += 1
    return freq_list / np.sum(freq_list)


def main():
    seq_dir = '/mnt/data/jiangyong/vito/offline_root'
    freq_record = 'freq_record.npy'
    if os.path.exists(freq_record):
        freq_list = np.load(freq_record)
    else:
        freq_list = index_freq(seq_dir, 1024)
        np.save('freq_record.npy', freq_list)
    plt.bar(np.arange(1, 1025), freq_list)
    plt.xlabel('index')
    plt.ylabel('frequency')
    plt.title('histogram of code frequency')
    max_freq = np.max(freq_list)
    upper_y1 = np.round(max_freq, decimals=2)
    upper_y1 = upper_y1 + 0.01 if upper_y1 < max_freq else upper_y1
    plt.ylim(0, upper_y1)
    plt.grid(True)
    plt.savefig('code_freq.png')
    plt.clf()

    embed_path = '/mnt/data/jiangyong/vito/vqgan_embed.pth'
    codebook = torch.load(embed_path, map_location='cpu').numpy()

    norm_list = np.linalg.norm(codebook, axis=1)
    plt.bar(np.arange(1, 1025), norm_list)
    plt.xlabel('index')
    plt.ylabel('L2-norm')
    plt.title('distribution of code vector norm')
    max_norm = np.max(norm_list)
    upper_y2 = np.round(max_norm)
    upper_y2 = upper_y2 + 1 if upper_y2 < max_norm else upper_y2
    plt.ylim(0, upper_y2)
    plt.grid(True)
    plt.savefig('code_norm.png')
    plt.clf()

    codebook_reduced = TSNE(n_components=2, learning_rate='auto').fit_transform(codebook)
    color = np.zeros((1024, 3))
    color[:, 0] = np.linspace(1, 0, 1024)
    color[:, 2] = np.linspace(0, 1, 1024)
    plt.scatter(codebook_reduced[:, 0], codebook_reduced[:, 1], s=1, c=color)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('t-SNE on codebook 256-d embedding space')
    plt.grid(False)
    plt.savefig('code_t-SNE.png')
    plt.clf()

    vocab_path = '/mnt/data/jiangyong/vito/vocab_embed.npy'
    vocab_embed = np.load(vocab_path)
    vocab_norm_list = np.linalg.norm(vocab_embed, axis=1)
    plt.bar(np.arange(1, len(vocab_norm_list)+1), vocab_norm_list)
    plt.xlabel('index')
    plt.ylabel('L2-norm')
    plt.title('distribution of vocab vector norm')
    max_norm_vocab = np.max(vocab_norm_list)
    upper_y3 = np.round(max_norm_vocab)
    upper_y3 = upper_y3 + 1 if upper_y3 < max_norm_vocab else upper_y3
    plt.ylim(0, upper_y3)
    plt.grid(True)
    plt.savefig('vocab_norm.png')
    plt.clf()


if __name__ == '__main__':
    main()
