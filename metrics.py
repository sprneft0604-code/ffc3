from cleanfid import fid

import numpy as np
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except Exception:
    from skimage.measure import compare_psnr as peak_signal_noise_ratio
    from skimage.measure import compare_ssim as structural_similarity
# from skimage.measure import compare_psnr, compare_ssim
import math
import sys
from skimage import io, color, filters
import os
import math
import torch
from torch.autograd import Variable
from torchvision.models.inception import inception_v3
from torch.nn import functional as F
from torch import nn
from scipy.stats import entropy
from torchvision import transforms
from PIL import Image
import torch.utils.data as data
from cleanfid.inception_pytorch import InceptionV3
from cleanfid.inception_torchscript import InceptionV3W

def rmetrics(a, b):
    import numpy as _np
    # 兼容新旧 skimage API
    try:
        from skimage.metrics import peak_signal_noise_ratio as _psnr_fn, structural_similarity as _ssim_fn
    except Exception:
        from skimage.measure import compare_psnr as _psnr_fn, compare_ssim as _ssim_fn

    a_arr = _np.asarray(a).astype(_np.float64)
    b_arr = _np.asarray(b).astype(_np.float64)

    # 确定 data_range：优先按像素范围判断（uint8 -> 255），否则用 max-min，保证 >0
    try:
        max_val = max(a_arr.max(), b_arr.max())
        min_val = min(a_arr.min(), b_arr.min())
        if max_val > 1.5:  # 很可能是 [0,255] 的 uint8
            data_range = 255.0
        else:
            data_range = float(max_val - min_val) if (max_val - min_val) > 0 else 1.0
    except Exception:
        data_range = 1.0

    # 计算 PSNR（兼容旧版函数签名）
    try:
        psnr = _psnr_fn(a_arr, b_arr, data_range=data_range)
    except TypeError:
        psnr = _psnr_fn(a_arr, b_arr)

    # 计算 SSIM（处理 multichannel / channel_axis 的 API 差异）
    try:
        ssim = _ssim_fn(a_arr, b_arr, multichannel=True, data_range=data_range)
    except TypeError:
        try:
            ssim = _ssim_fn(a_arr, b_arr, channel_axis=-1, data_range=data_range)
        except TypeError:
            # 退回不带额外参数的调用（最不优雅但可兼容）
            ssim = _ssim_fn(a_arr, b_arr)

    return psnr, ssim

def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    #2nd term
    top = np.int32(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int32(al1 * len(rgl))
    T2 = np.int32(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm,uciqe

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)

            m = top/bottom
            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)

def cal_ssim_psnr(reference_path, result_path):
    """
    对齐尺寸并计算平均 PSNR/SSIM。
    说明：
    - 自动尝试把生成文件名的 'Out_' 前缀去掉以匹配参考文件名。
    - 若尺寸不同，会把生成图像 resize 到参考图像尺寸（BICUBIC）。
    - 会跳过无法匹配到参考图像的生成文件并打印 warning。
    """
    result_files = sorted(os.listdir(result_path))
    # 预列参考目录下的文件（若是目录）
    ref_files = []
    if os.path.isdir(reference_path):
        ref_files = sorted(os.listdir(reference_path))

    sumpsnr, sumssim = 0.0, 0.0
    N = 0

    for fname in result_files:
        if not any(fname.lower().endswith(ext) for ext in IMG_EXTENSIONS):
            continue

        gen_path = os.path.join(result_path, fname)

        # 构造候选参考名
        candidates = [fname]
        if fname.startswith('Out_'):
            candidates.append(fname[len('Out_'):])
        if 'corrected' in fname:
            candidates.append(fname.replace('corrected', ''))
        candidates = [c.lstrip('_') for c in candidates]

        # 尝试找到参考路径
        ref_path = None
        for cand in candidates:
            cand_path = os.path.join(reference_path, cand)
            if os.path.exists(cand_path):
                ref_path = cand_path
                break

        # 模糊匹配（后缀/包含）
        if ref_path is None and ref_files:
            for rf in ref_files:
                for cand in candidates:
                    if rf == cand or rf.endswith(cand) or cand.endswith(rf):
                        ref_path = os.path.join(reference_path, rf)
                        break
                if ref_path:
                    break

        if ref_path is None:
            print('Warning: reference not found for generated file:', fname, 'candidates:', candidates)
            continue

        # 读取并统一为 RGB，若尺寸不同则 resize 生成图到参考尺寸
        try:
            gen_img = Image.open(gen_path).convert('RGB')
            ref_img = Image.open(ref_path).convert('RGB')
        except Exception as e:
            print('Warning: failed to open image pair:', gen_path, ref_path, e)
            continue

        if gen_img.size != ref_img.size:
            gen_img = gen_img.resize(ref_img.size, Image.BICUBIC)

        gen_arr = np.asarray(gen_img).astype(np.float64)
        ref_arr = np.asarray(ref_img).astype(np.float64)

        try:
            psnr, ssim = rmetrics(gen_arr, ref_arr)
        except Exception as e:
            print('Warning: failed to compute metrics for pair:', gen_path, ref_path, e)
            continue

        sumpsnr += psnr
        sumssim += ssim
        N += 1

    if N == 0:
        print('No matched image pairs found for PSNR/SSIM calculation.')
        return 0.0, 0.0

    mpsnr = sumpsnr / N
    mssim = sumssim / N
    return mpsnr, mssim
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class BaseDataset(data.Dataset):
    def __init__(self, data_root, image_size=[256, 256], loader=pil_loader):
        self.imgs = make_dataset(data_root)
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        return img

    def __len__(self):
        return len(self.imgs)

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    # inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    # inception_model.eval()

    inception_model = InceptionV3W('./', download=True, resize_inside=False).to(0)
    inception_model.eval()

    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 2048))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)
if __name__ == '__main__':
    target = 'target_dir'
    output = 'output_dir'
    fid_score = fid.compute_fid(target, output, mode='clean', num_workers=0, batch_size=64)
    mpsnr, mssim = cal_ssim_psnr(target, output)
    is_score = inception_score(BaseDataset(output), cuda=True, batch_size=8, resize=True, splits=10)
    print('{}, fid = {}, is={} psnr = {}, ssim = {}'.format(output, fid_score, is_score, mpsnr, mssim))