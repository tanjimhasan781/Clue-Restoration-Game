import numpy as np

def apply_convolution_2d(image, kernel):
    img_height,img_width=image.shape
    kernel_height,kernel_width=kernel.shape

    pad_h=kernel_height//2
    pad_w=kernel_width//2
    
    padded_image=np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    output=np.zeros_like(image,dtype=np.float32)
    
    for i in range(img_height):
        for j in range(img_width):
            region=padded_image[i:i+kernel_height,j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    output=np.clip(output, 0, 255)
    return output.astype(np.uint8)

def apply_convolution(image, kernel):
    if len(image.shape)==3:
        output=np.zeros_like(image)
        for c in range(image.shape[2]):
            output[:, :, c]=apply_convolution_2d(image[:, :, c].astype(np.float32), kernel)
        return output
    else:
        return apply_convolution_2d(image.astype(np.float32),kernel)


def Gaussian_Smoothing_Function(u,v,sigma):
    return (1/(2*np.pi*sigma**2))*np.exp(-(u**2+v**2)/(2*sigma**2))


def create_gaussian_kernel(kernel_size, sigma):
    if isinstance(kernel_size, tuple):
        kernel_size=kernel_size[0]
    k=kernel_size//2
    kernel=np.zeros((kernel_size,kernel_size ), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            u = i-k
            v = j-k 
            kernel[i,j] = Gaussian_Smoothing_Function(u,v,sigma)
    kernel /= np.sum(kernel)
    
    return kernel


def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=1.5):
    if image is None:
        return None
    
    kernel=create_gaussian_kernel(kernel_size, sigma)
    smooth_image=apply_convolution(image,kernel)
    
    return smooth_image

def apply_median_filter_2d(channel, kernel_size, pad):
    height, width = channel.shape
    padded = np.pad(channel, pad, mode='edge')
    output = np.zeros_like(channel)
    
    for i in range(height):
        for j in range(width):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(window)
    
    return output.astype(np.uint8)


def apply_median_filter(image, kernel_size=5):
    if image is None:
        return None
    
    pad = kernel_size//2
    if len(image.shape) == 3:
        output = np.zeros_like(image)
        for c in range(image.shape[2]):
            output[:, :, c] = apply_median_filter_2d(image[:, :, c], kernel_size, pad)
        return output
    else:
        return apply_median_filter_2d(image, kernel_size, pad)

def apply_bilateral_filter_2d(channel, radius, sigma_color, sigma_space):
    height, width = channel.shape
    padded = np.pad(channel, radius, mode='edge')
    output = np.zeros_like(channel, dtype=np.float32)
    
    spatial_gauss = np.zeros((2*radius+1, 2*radius+1))
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            spatial_gauss[i+radius, j+radius] = np.exp(-(i**2 + j**2) / (2 * sigma_space**2))
    
    for i in range(height):
        for j in range(width):
            window = padded[i:i+2*radius+1, j:j+2*radius+1].astype(np.float32)
            center_value = float(channel[i, j])
            intensity_diff = window - center_value
            intensity_gauss = np.exp(-(intensity_diff**2) / (2 * sigma_color**2))
            weights = spatial_gauss * intensity_gauss
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                output[i, j] = np.sum(window * weights) / weights_sum
            else:
                output[i, j] = center_value
    
    return output

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    if image is None:
        return None
    
    radius=d//2
    if len(image.shape) == 3:
        output = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            output[:, :, c] = apply_bilateral_filter_2d(image[:, :, c], radius, sigma_color, sigma_space)
        return output.astype(np.uint8)
    else:
        return apply_bilateral_filter_2d(image, radius, sigma_color, sigma_space).astype(np.uint8)

def apply_unsharp_mask(image, sigma=2.0, amount=0.5):
    if image is None:
        return None
    
    kernel_size=5
    blurred=apply_gaussian_filter(image,kernel_size=(kernel_size, kernel_size),sigma=sigma)
    
    original_float=image.astype(np.float32)
    blurred_float=blurred.astype(np.float32)
    sharpened=original_float+amount*(original_float-blurred_float)
    sharpened=np.clip(sharpened, 0, 255)

    return sharpened.astype(np.uint8)

def LoG_Function(u,v,sigma):
    return (-1/(np.pi*sigma**4))*(1-(u**2+v**2)/(2*sigma**2)) * np.exp(-(u**2+v**2)/(2*sigma**2))

def create_LoG_kernel(sigma):
    size=int(9*sigma)
    if size%2==0:
        size+=1

    k=size//2
    kernel=np.zeros((size,size ),dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u=i-k
            v=j-k 
            kernel[i,j]=LoG_Function(u,v,sigma)
    kernel=kernel-kernel.mean() 
    return kernel 

def apply_LoG_sharpen(image,sigma):
    if image is None:
        return None
    
    log_kernel=create_LoG_kernel(sigma=sigma)
    sharpened=apply_convolution(image,log_kernel)
    
    return sharpened

def apply_gamma_correction(image, gamma=2.2):
    if image is None:
        return None
    
    inv_gamma=1.0/gamma
    table=np.array([((i/255.0)**inv_gamma)*255 
                      for i in range(256)]).astype(np.uint8)
    
    corrected=table[image]
    
    return corrected

def apply_contrast_stretch(image, alpha=1.5, beta=30):
    if image is None:
        return None
    
    stretched = image.astype(np.float32)
    stretched = alpha * stretched + beta
    stretched = np.clip(stretched, 0, 255)
    
    return stretched.astype(np.uint8)
