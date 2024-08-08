import numpy as np
import torch

def rle_to_mask(rle_string,height,width):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rleString (str): Description of arg1 
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    torch.Tensor: torch tensor of the mask
    '''
    rows, cols = height, width
    if rle_string == '-1':
        return torch.zeros((height, width))
    else:
        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 1
        img = img.reshape(cols,rows)
        img = img.T
        return torch.from_numpy(img)

def mask_to_rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    img = img.cpu().numpy()
    pixels= np.round(img.T.flatten()).astype(int)
    if np.all(np.allclose(pixels, 0)):
        return '-1' # for compatibility
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        
    if len(runs) % 2 == 1:
        runs = runs[:-1]
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)