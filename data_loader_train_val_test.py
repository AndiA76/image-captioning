import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/home/andreas/data/coco/'): # Change this path to where the coco data is stored
    """Returns a data loader for image captioning on coco dataset 2014. It can be used in 4 different modes:
    Modes:            
      mode = 'train':      Returns a data loader for mini-batch training, which provides batches of transformed and 
                           ormalized images with ground truth captions of the same length from training dataset
      mode = 'batch-val':  Returns a data loader for mini-batch validation in parallel to the training process,  
                           which provides batches of transformed and normalized images with ground truth captions 
                           of the same length from validation dataset
      mode = 'single-val': Returns a data loader for single-step validation (for batch_size == 1), which provides 
                           single images as unresized original resp. transformed and normalized version with ground 
                           truth caption from validation dataset
      mode = 'test':       Returns a data loader for single-step testing (for batch_size == 1) on new test images 
                           without any captions, which provides single images as unresized resp. transformed and 
                           normalized version from test dataset (ground truth captions are not provided)
    Args:
      transform:           Image transform.
      mode:                One of 'train', 'batch-val', 'single-val' or 'test'.
      batch_size:          Batch size (batch_size >= 1 if in training or batch-validation mode; batch_size == 1 if 
                           in single-step-validation or testing mode).
      vocab_threshold:     Minimum word count threshold. Words which occur more seldom in training vocabulary than 
                           this minimum threshold are considered unknown words.
      vocab_file:          File containing the vocabulary. 
      start_word:          Special word denoting sentence start.
      end_word:            Special word denoting sentence end.
      unk_word:            Special word denoting unknown words.
      vocab_from_file:     If False, create vocab from scratch & override any existing vocab_file.
                           If True, load vocab from from existing vocab_file, if it exists.
      num_workers:         Number of subprocesses to use for data loading 
      cocoapi_loc:         The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    # Check mode
    print('Data loader called in {}-mode.'.format(mode))
    assert mode in ['train', 'batch-val', 'single-val', 'test'], "mode must be one of 'train', 'batch-val', 'single-val' or 'test'."
    # Vocabulary shall only be generated in training mode
    if vocab_from_file==False: 
        assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."
    
    # Based on mode ['train', 'batch-val', 'single-val', 'test'], obtain img_folder and annotations_file.
    if mode == 'train':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, 'images/train2014/')
        annotations_file = os.path.join(cocoapi_loc, 'annotations/captions_train2014.json')
    
    if mode == 'batch-val' or mode == 'single-val':
        if mode == 'single-val':
            assert batch_size==1, "Please change batch_size to 1 if validating your model in single-step mode."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'images/val2014/')
        annotations_file = os.path.join(cocoapi_loc, 'annotations/captions_val2014.json')
    
    if mode == 'test':
        assert batch_size==1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'images/test2014/')
        annotations_file = os.path.join(cocoapi_loc, 'annotations/image_info_test2014.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == 'train' or mode == 'batch-val':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    
    # else if mode == 'single-val' or mode == 'test'
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      num_workers=num_workers,
                                      batch_size=dataset.batch_size,
                                      shuffle=True)
        
    # Return data loader depending on the selected mode
    return data_loader


class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        
        # Training mode or batch-validation mode
        if self.mode == 'train' or self.mode == 'batch-val':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        # Single-validation mode
        elif self.mode == 'single-val':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
            # Add annotations to image files
            val_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in val_info['images']]
        # Testing mode
        else:
            # Add annotations ot image files
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
        
    def __getitem__(self, index):
        # Obtain transformed image with caption if in training or batch-validation mode
        if self.mode == 'train' or self.mode == 'batch-val':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            
            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)
            
            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()
            
            # return pre-processed image and caption tensors
            return image, caption
        
        # Obtain original and transformed image with caption if in single-step validation mode
        elif self.mode == 'single-val':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            
            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)
            
            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()
            
            # return unresized original image, pre-processed image and caption tensors (batch_size == 1)
            return orig_image, image, caption
            
        # Obtain original and transformed image if in testing mode
        else:
            path = self.paths[index]
            
            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)
            
            # return unresized original image and pre-processed image tensor (batch_size == 1)
            return orig_image, image

    def get_train_indices(self):
        # Randomly sample batch indices for images with captions from training dataset (when in 'train'-mode) or 
        # from validation dataset (when in 'batch-validation'-mode), where all image captions have the same length
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices
        
    def __len__(self):
        if self.mode == 'train' or self.mode == 'batch-val':
            # Return length of available annotation keys
            return len(self.ids)
        else:
            # Return length of available image paths in testing or validation dataset
            return len(self.paths)