import argparse
import os
import subprocess
import numpy as np
import os.path as osp
import sys
from imageio import imwrite
import pandas as pd
import random
import cv2

mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

def check_mnist_dir(data_dir):

    downloaded = np.all([osp.isfile(osp.join(data_dir, key)) for key in mnist_keys])
    if not downloaded:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_mnist(data_dir)
    else:
        print('MNIST was found')
        
def download_mnist(data_dir):

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    for k in mnist_keys:
        k += '.gz'
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gunzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)
        
def extract_mnist(data_dir):

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1))

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape((num_mnist_train)))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1))

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape((num_mnist_test)))

    """
    return np.concatenate((train_image, test_image)), \
        np.concatenate((train_label, test_label))
    """
    return train_image, train_label, test_image, test_label

def sample_coordinate(high, size):
    if high > 0:
        return np.random.randint(high, size=size)
    else:
        return np.zeros(size).astype(np.int)

def multi_mnist_generator(config):
    # extract mnist images and labels
    train_image, train_label, test_image, test_label = extract_mnist(config.mnist_path)
    h, w = train_image.shape[1:3]
    
    # label index
    train_indexes = []
    for c in range(10):
        train_indexes.append(list(np.where(train_label == c)[0]))
        
    test_indexes = []
    for c in range(10):
        test_indexes.append(list(np.where(test_label == c)[0]))
    
    if not os.path.exists(config.multimnist_path):
        os.makedirs(config.multimnist_path)
    
    for i, split_name in enumerate(['train', 'val', 'test']):
        path = osp.join(config.multimnist_path, split_name)
        print('Generate images for {} at {}'.format(split_name, path))
        if not os.path.exists(path):
            os.makedirs(path)
        labels = []
        for j in range(100):
            current_class = str(j).zfill(2)
            class_path = osp.join(path, current_class)
            print('{} (progress: {}/{})'.format(class_path, j + 100*i, 300))
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                
            for k in range(config.num_image_per_train_val_test_class[i]):
                # sample images
                digits = [int(char) for char in current_class]
                if i == 0:
                    imgs = [np.squeeze(train_image[np.random.choice(train_indexes[d])]) for d in digits]
                else:
                    imgs = [np.squeeze(test_image[np.random.choice(test_indexes[d])]) for d in digits]
                background = np.zeros((64, 64)).astype(np.uint8)
                
                # make label
                file_name = '{}_{}.png'.format(k, current_class)
                file_path = osp.join(class_path, file_name)
                label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, current_class, file_path]
                
                for a in range(len(digits)):
                    label[digits[a]] += 0.5
                    
                labels.append(label)
                
                # sample coordinates
                ys = sample_coordinate(64-h, 2)
                xs = sample_coordinate(64//2-w, 2)
                xs = [l*64//2 + xs[l]
                      for l in range(2)]
                # combine images
                for l in range(2):
                    background[ys[l]:ys[l]+h, xs[l]:xs[l]+w] = imgs[l]
                # write the image
                imwrite(file_path, background)
        df = pd.DataFrame(
            labels, columns = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9', 'class', 'file_path']  
        )
        csv_file_name = split_name + '.csv'
        csv_file_path = osp.join(config.multimnist_path, csv_file_name)
        df.to_csv(csv_file_path)
    
def contrast_mnist_generator(config):
    # extract mnist images and labels
    train_image, train_label, test_image, test_label = extract_mnist(config.mnist_path)
    h, w = train_image.shape[1:3]
    
    # label index
    train_indexes = []
    for c in range(10):
        train_indexes.append(list(np.where(train_label == c)[0]))
        
    test_indexes = []
    for c in range(10):
        test_indexes.append(list(np.where(test_label == c)[0]))
    
    if not os.path.exists(config.multimnist_path):
        os.makedirs(config.multimnist_path)
        
    for i, split_name in enumerate(['train', 'val', 'test']):
        path = osp.join(config.multimnist_path, split_name)
        print('Generate images for {} at {}'.format(split_name, path))
        if not os.path.exists(path):
            os.makedirs(path)
        labels = []
        for j in range(100):
            current_class = str(j).zfill(2)
            class_path = osp.join(path, current_class)
            print('{} (progress: {}/{})'.format(class_path, j + 100*i, 300))
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            for k in range(config.num_image_per_train_val_test_class[i]):
                # sample images
                digits = [int(char) for char in current_class]
                if i == 0:
                    imgs = [np.squeeze(train_image[np.random.choice(train_indexes[d])]) for d in digits]
                else:
                    imgs = [np.squeeze(test_image[np.random.choice(test_indexes[d])]) for d in digits]
                background = np.zeros((64, 64)).astype(np.uint8)
                
                # generate contrast
                contrast = []
                contrast.append(random.random())
                contrast1 = 1 - contrast[0]
                contrast.append(contrast1)
                
                # make label
                file_name = '{}_{}.png'.format(k, current_class)
                file_path = osp.join(class_path, file_name)
                label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, current_class, file_path]
                
                label[digits[0]] = contrast[0]
                label[digits[1]] += contrast[1]
                    
                labels.append(label)
                
                # sample coordinates
                ys = sample_coordinate(64-h, 2)
                xs = sample_coordinate(64//2-w, 2)
                xs = [l*64//2 + xs[l]
                      for l in range(2)]
                # combine images
                for l in range(2):
                    imgs[l] = imgs[l]*contrast[l]
                    background[ys[l]:ys[l]+h, xs[l]:xs[l]+w] = imgs[l]
                # write the image
                imwrite(file_path, background)
        df = pd.DataFrame(
            labels, columns = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9', 'class', 'file_path']  
        )
        csv_file_name = split_name + '.csv'
        csv_file_path = osp.join(config.multimnist_path, csv_file_name)
        df.to_csv(csv_file_path)
        
def add_padding(image, padding_size):
    height, width = image.shape[:2]

    # 中央に配置して黒い余白を追加
    padded_image = cv2.copyMakeBorder(image, 0, padding_size, 0, padding_size,
                                      cv2.BORDER_CONSTANT, value=[0])
    return padded_image

def remove_padding(image, padding_size):
    # 中央の位置を計算
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    height, width = image.shape[:2]
    # 中央を基準にして黒い余白を削除
    cropped_image = image[:height - padding_size, :width - padding_size]
    return cropped_image
    
def resize_image(image, resized_size):
    # 画像の高さと幅を取得
    # 倍率によって余白を追加、削除する
    height, width = image.shape[:2]
    scale_factor = resized_size / 28
    
    if scale_factor > 1:
        image = add_padding(image, resized_size - 28)
        height, width = image.shape[:2]
        transformation_matrix = np.array([[scale_factor, 0, 0],
                                       [0, scale_factor, 0]], dtype=np.float32)
        # アフィン変換を適用
        resized_image = cv2.warpAffine(image, transformation_matrix, (width, height))
    else:
        # アフィン変換の変換行列を生成
        transformation_matrix = np.array([[scale_factor, 0, 0],
                                       [0, scale_factor, 0]], dtype=np.float32)
        # アフィン変換を適用
        resized_image = cv2.warpAffine(image, transformation_matrix, (width, height))
        
        # 余白を削除
        resized_image = remove_padding(resized_image, 28 - resized_size)
    
    height, width = resized_image.shape[:2]

    return resized_image
    
def scaled_mnist_generator(config):
    # extract mnist images and labels
    train_image, train_label, test_image, test_label = extract_mnist(config.mnist_path)
    h, w = train_image.shape[1:3]
    
    # label index
    train_indexes = []
    for c in range(10):
        train_indexes.append(list(np.where(train_label == c)[0]))
        
    test_indexes = []
    for c in range(10):
        test_indexes.append(list(np.where(test_label == c)[0]))
    
    if not os.path.exists(config.multimnist_path):
        os.makedirs(config.multimnist_path)
    
    for i, split_name in enumerate(['train', 'val', 'test']):
        path = osp.join(config.multimnist_path, split_name)
        print('Generate images for {} at {}'.format(split_name, path))
        if not os.path.exists(path):
            os.makedirs(path)
        labels = []
        for j in range(100):
            current_class = str(j).zfill(2)
            class_path = osp.join(path, current_class)
            print('{} (progress: {}/{})'.format(class_path, j + 100*i, 300))
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                
            for k in range(config.num_image_per_train_val_test_class[i]):
                # sample images
                digits = [int(char) for char in current_class]
                if i == 0:
                    imgs = [np.squeeze(train_image[np.random.choice(train_indexes[d])]) for d in digits]
                else:
                    imgs = [np.squeeze(test_image[np.random.choice(test_indexes[d])]) for d in digits]
                background = np.zeros((64, 64)).astype(np.uint8)
                
                # generate scale ratio
                rng = np.random.default_rng()
                digit_sizes = []
                while True:
                    digit_sizes.append(rng.random())
                    if (digit_sizes[0] != 0.0) & (digit_sizes != 1.0):
                        break
                    else:
                        digit_sizes.clear()
                digit_sizes.append(1.0 - digit_sizes[0])
                
                # make label
                file_name = '{}_{}.png'.format(k, current_class)
                file_path = osp.join(class_path, file_name)
                label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, current_class, file_path]
                
                label[digits[0]] = digit_sizes[0]
                label[digits[1]] += digit_sizes[1]
                    
                labels.append(label)
                
                sizes = [int(64*digit_sizes[0]), 64 - int(64*digit_sizes[0])]
                
                if sizes[0] == 64:
                    sizes[0] = 63
                    sizes[1] = 1
                elif sizes[0] == 0:
                    sizes[0] = 1
                    sizes[1] = 63
                
                for a in range(2):
                    imgs[a] = resize_image(imgs[a], sizes[a])

                # sample coordinates
                y_0 = int(sample_coordinate(64-sizes[0], 1))
                # x_0 = int(sample_coordinate(sizes[0], 1))
                x_0 = 0
                y_1 = int(sample_coordinate(64-sizes[1], 1))
                # x_1 = int(sample_coordinate(sizes[1], 1) + sizes[0])
                x_1 = 0 + sizes[0]
                
                # combine images
                background[y_0:y_0+sizes[0], x_0:x_0+sizes[0]] = imgs[0]
                background[y_1:y_1+sizes[1], x_1:x_1+sizes[1]] = imgs[1]
                
                # write the image
                imwrite(file_path, background)
        df = pd.DataFrame(
            labels, columns = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9', 'class', 'file_path']  
        )
        csv_file_name = split_name + '.csv'
        csv_file_path = osp.join(config.multimnist_path, csv_file_name)
        df.to_csv(csv_file_path)

def generator(config):
    # check if mnist is downloaded. if not, download it
    check_mnist_dir(config.mnist_path)
    
    if config.mode == 'MultiMNIST':
        multi_mnist_generator(config)
    elif config.mode == 'ContrastMultiMNIST':
        contrast_mnist_generator(config)
    elif config.mode == 'ScaledMultiMNIST':
        scaled_mnist_generator(config)
    else:
        sys.exit("Error: invalid mode")
    

def argparser():
    def str2bool(v):
        return v.lower() == 'true'
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mnist_path', type=str, default='./datasets/mnist/',
                        help='path to *.gz files')
    parser.add_argument('--multimnist_path', type=str, default='./datasets/multimnist')
    parser.add_argument('--num_image_per_train_val_test_class', type=int, nargs='+', default=[1000, 100, 100])
    parser.add_argument('--mode', type=str, default='MultiMNIST')
    parser.add_argument('--random_seed', type=int, default=123)
    config = parser.parse_args()
    return config

def main():
    config = argparser()
    assert len(config.num_image_per_train_val_test_class) == 3
    generator(config)

if __name__ == '__main__':
    main()
    