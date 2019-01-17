import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

# for plotting, visualizing, debugging
from PIL import Image
from livelossplot import PlotLosses
from IPython import display
import matplotlib.pyplot as plt

def plot_tensor_image(ax, tensor, title):
    """
    """
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image[image>1] = 1
    image[image<0] = 0
    to_pil = transforms.ToPILImage()
    image = to_pil(image)
    out = ax.imshow(image)
    ax.set_title(title)
    return out

# Constants
# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gatys model constants
mean = [0.40760392, 0.45795686, 0.48501961] #BGR
std = [1,1,1]
r_std = [1/el for el in std]
r_mean = [-1 * el/std[i] for i,el in enumerate(mean)]

# NORMALIZATION AND RESIZING
testing = True
preproc = [transforms.ToTensor(), 
           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
           transforms.Normalize(mean=mean, std=[1,1,1]),
           transforms.Lambda(lambda x: x.mul_(255))]

if testing:
    preproc = [transforms.Resize(256)] + preproc # 

postproc = [transforms.Lambda(lambda x: x.mul(1/255)), 
            transforms.Normalize(mean=r_mean, std=r_std),
            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])] #turn to RGB
    
loader = transforms.Compose(preproc)  
unloader = transforms.Compose(postproc)

def image_loader(image_path):
    image = Image.open(image_path)
#     print(image.mode)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    # drop the 4th channel if it exists
    if image.shape[1] != 3:
        print('image shape is {0}, (batch, channel, HxW)'.format(image.shape))
        print('taking first 3 channels, assuming 4th is alpha (transparency)')
        image = image[:,:3,:,:]
        
    return image.to(device, torch.float)

def imshow(tensor, title=None, update=False):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image[image>1] = 1
    image[image<0] = 0
    to_pil = transforms.ToPILImage()
    image = to_pil(image)
    _ = plt.imshow(image)
    if update:
        display.display(_)
        display.clear_output(wait=True)
    if title is not None:
        plt.title(title)
    plt.pause(0.000001) # pause a bit so that plots are updated
    return image

# Gatys VGG class
class VGG_g(nn.Module):
    def __init__(self, pool='max'):
        super(VGG_g, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
    
# Mine
def calc_2_moments(x):
    # VERIFIED THAT IT MATCHES NUMPY np.cov
    a, b, c, d = x.size()  # bs, channels/featuremaps, h, w
    features = x.view(a * b, c * d)
    mean = features.mean(dim=1).view(b,1)
    cov = torch.matmul((features-mean),(features-mean).t()).div(c*d)
    return mean, cov

def calc_eigstuff(cov, clamp=True):
    eival, eivec = torch.symeig(cov, eigenvectors=True)
#     import ipdb; ipdb.set_trace()
    # torch.clamp zeros negative eigenvalues so torch.sqrt doesn't give nans
    if clamp:
        # How does clamping (not)affect the root_cov calculation?
        # Answer: Because it was supposed to be an extremely small positive
        # square root number calculated, so small that numerical instability
        # made it negative. So we just treat it as 0.
        # if A is nxn, symmetric matrix (like cov)
        # A = V@D@Vinv, let R = principal square root of A (such that R@R=A)
        # then R = V@S@Vinv, where S is ONE of many square roots of D
        # since there are two possible choices for a square root of each
        # diagonal element of D, there are 2n choices for the matrix D½.
        # We choose to take the all positive root of D
        # When you look at the eigenvals, the negatives are all to ne10**-8
        # or so they are all supposed to be near 0. So we clamp.
        c_eival = torch.clamp(eival, min=0.000001**2)
        sqrt_eival = torch.diag(torch.sqrt(c_eival)) 
#         sqrt_eival.requires_grad_()
#         sqrt_eival.retain_graph()
        #torch.matmul(torch.eye(len(torch.sqrt(c_eival))).cuda(), torch.sqrt(c_eival))
    else:
        print('never actually not clamp. We clamp because of num instability.')
        c_eival = eival
        sqrt_eival = torch.diag(torch.sqrt(eival))
    # since cov is symmetric, inverse == transpose
    # wanted to use inverse since by inspection seemed more precision
    # but throws singular matrix error sometimes when it shouldn't. use .t()
    root_cov = torch.matmul(torch.matmul(eivec,sqrt_eival), eivec.t())
    # trace is sum of diagonals, eival would be multiplied by I, so tr == sum
    tr_cov = torch.sum(eival)
#     assert tr_cov >= 0
#     import ipdb; ipdb.set_trace()
    return tr_cov, root_cov

def calc_l2wass_dist(mean_synth, cov_synth, layer_style_desc):
    mean_stl, tr_cov_stl, root_cov_stl = layer_style_desc
    
    tr_cov_synth = torch.trace(cov_synth)
    
    mean_diff_squared = torch.sum(torch.pow((mean_synth - mean_stl),2))
    
    cov_prod = torch.matmul(torch.matmul(root_cov_stl, cov_synth), root_cov_stl)
    
    _, root_cov_prod = calc_eigstuff(cov_prod)
    
    tr_root_cov_prod = torch.trace(root_cov_prod)
        
    # numerical inaccuracies, sometimes could be trying to take sqrt of negative. take abs
    dist = torch.sqrt(torch.abs(mean_diff_squared + tr_cov_synth + tr_cov_stl - (2*tr_root_cov_prod)))
#     dist[torch.isnan(dist)] = 0
#     assert torch.isnan(dist).item() == False
    return dist

def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a*b, c*d)
    G = torch.matmul(features, features.t()).div(c*d)
    return G

def gram_loss(gram_target, input):
    return F.mse_loss(gram_target, gram_matrix(input))

def get_input_optimizer(input_img, opt_type):
    # this line to show that input is a parameter that requires a gradient
    if opt_type == 'lbfgs':
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    elif opt_type == 'adam':
        optimizer = optim.Adam([input_img.requires_grad_()], lr=5e0)
    return optimizer    

#vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, base_cnn, use_lyrs, pool_type):
        super(VGG, self).__init__()
        self.use_lyrs, self.pool_type = use_lyrs, pool_type
        self.lyrs = {}
        for name in self.use_lyrs:
            if name[:4] == 'conv':
                self.lyrs[name] = getattr(base_cnn, name)
            elif name[:4] == 'pool':
                self.lyrs[name] = self.pool_func(self.pool_type)
            elif name[:4] == 'relu':
                self.lyrs[name] = nn.ReLU()
            else:
                raise RuntimeError('Unknow layer type')
        
    def forward(self, x):
        out = {}
        for i, cur_lyr_name in enumerate(self.use_lyrs):
            if i == 0:
                out[cur_lyr_name] = self.lyrs[cur_lyr_name](x)
            else:
                prev_lyr_name = self.use_lyrs[i-1]
                out[cur_lyr_name] = self.lyrs[cur_lyr_name](out[prev_lyr_name])
        return out
    
    def pool_func(self, pool_type):
        kern_sz = 2
        if pool_type=='avg':
            pool = torch.nn.AvgPool2d(kern_sz)
        elif pool_type=='max':
            pool = nn.MaxPool2d(kern_sz)
        return pool 
    
class Downsize(nn.Module):
    def __init__(self, h, w):
        super(Downsize, self).__init__()
        self.h, self.w = h,w
        
    def forward(self, img):
        m = nn.AdaptiveAvgPool2d((self.h, self.w))
        return m(img)

class TransferStyle(object):
    '''approx_stl is WIP'''
    def __init__(self, base_cnn, cont_loss_type, stl_loss_type, cont_img, stl_img, pool_type, cont_lyrs, stl_lyrs, device, resize_conv, downsample, approx_stl):
        self.all_lyr_names = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        ]
        self.base_cnn = base_cnn #copy.deepcopy(base_cnn)
        self.cont_loss_type = cont_loss_type
        self.stl_loss_type = stl_loss_type
        self.cont_img = cont_img.detach()
        self.stl_img = stl_img.detach()
        self.pool_type = pool_type
        self.cont_lyrs, self.stl_lyrs = cont_lyrs, stl_lyrs
        self.device = device
        self.resize_conv, self.downsample = resize_conv, downsample
        self.approx_stl = approx_stl
        self.cnn = self.construct_cnn()
        self.cont_targets  = self.construct_cont_targets()
        self.stl_targets = self.construct_stl_targets()
        # set up style weights
        self.get_style_weights()
            
    def construct_cnn(self):
        # have default layers if none are passed
        if not self.cont_lyrs: self.cont_lyrs = ['relu3_1']
        if not self.stl_lyrs: self.stl_lyrs = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1']
        max_cont_i = max([self.all_lyr_names.index(l) for l in self.cont_lyrs])
        max_stl_i = max([self.all_lyr_names.index(l) for l in self.stl_lyrs])
        self.use_lyrs = self.all_lyr_names[:max(max_cont_i, max_stl_i)+1]
        return VGG(self.base_cnn, self.use_lyrs, self.pool_type)

    def content_target(self, lyr, out):
        if self.cont_loss_type == 'mse':
            return out[lyr].detach()
        else:
            raise RuntimeError("cont_loss_type must be mse.")        
        
    def construct_cont_targets(self):
        out = self.cnn(self.cont_img)
        return {lyr: self.content_target(lyr, out) for lyr in self.cont_lyrs}
            
    def style_target(self, lyr, out):
        if self.stl_loss_type == 'gram':
            return gram_matrix(out[lyr].detach())
        elif self.stl_loss_type == 'l2':
            target = out[lyr].detach()
            if self.approx_stl:
                target = self.down_image(target)
            mean, cov = calc_2_moments(target)
            tr_cov, root_cov = calc_eigstuff(cov)
            return (mean, tr_cov, root_cov)
        else:
            raise RuntimeError('stl_loss_type must be "gram" or "l2".')            
                        
    def construct_stl_targets(self):
        out = self.cnn(self.stl_img)
        return {lyr: self.style_target(lyr, out) for lyr in self.stl_lyrs}
    
    def get_style_weights(self):
        if self.stl_loss_type == 'gram':
            # gatys defaults
            if self.stl_lyrs == ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']:
                self.stl_weights = [1e3/n**2 for n in [64,128,256,512,512]] 
            else:
                self.stl_weights = [1e1/(1+(.5*n))**2 for n in range(1,len(self.stl_lyrs)+1)] # untested
        elif self.stl_loss_type == 'l2':
            self.stl_weights = [1 for n in range(len(self.stl_lyrs))] # untested
            
    def content_loss(self, synth, target):
        if self.cont_loss_type == 'mse':
            return F.mse_loss(synth, target)

    def down_image(self, img):
        bs,c,h,w = img.size()
        points = h*w 
        scale_factor = 10000/points
        scale_factor = min(1, scale_factor)
        new_h, new_w = int(scale_factor*h), int(scale_factor*w)
        if new_h > 100 & new_w > 100:
            m = Downsize(new_h, new_w)
            return m(img)
        else:
            return img
        
    def style_loss(self, synth, target):
        if self.stl_loss_type == 'gram':
            return gram_loss(target, synth)
        elif self.stl_loss_type == 'l2':
            if self.approx_stl:
                synth = self.down_image(synth)
            mean, cov = calc_2_moments(synth)
            return calc_l2wass_dist(mean, cov, target)            
            
    def run_style_transfer(self, synth, opt_type, num_steps, stl_w, style_off = False, content_off = False, show_step = 25, loss_plots=True, show_synth=True, ax_synth = None, fig = None):
        optimizer = get_input_optimizer(synth, opt_type)
        model, cont_targets, stl_targets = self.cnn, self.cont_targets, self.stl_targets
        
        if loss_plots:
            liveloss = PlotLosses()
        run = [0]
        while run[0] <= num_steps:
            logs = {}
            
            def closure():
                optimizer.zero_grad()
                out = model(synth)

                if not style_off:
                    style_losses = [self.stl_weights[i] * self.style_loss(out[st], stl_targets[st]) for i, st in enumerate(stl_targets.keys())]
                    style_score = sum(style_losses)
                else:
                    style_score = 0
                    
                if not content_off:                    
                    content_losses = [self.content_loss(out[ct], cont_targets[ct]) for ct in cont_targets.keys()]
                    content_score = sum(content_losses)
                else:
                    content_score = 0
                    
                style_score *= stl_w
                loss = content_score + style_score
                loss.backward()
                run[0] += 1
                
                if loss_plots:
                    if not content_off:
                        logs['content_loss'] = content_score.item()
                    else:
                        logs['content_loss'] = content_score
                    if not style_off:
                        logs['style_loss'] = style_score.item()
                    else:
                        logs['style_loss'] = style_score
                    logs['total_loss'] = loss.item()

                return loss

            optimizer.step(closure)
            
            if loss_plots:
                liveloss.update(logs)
                
            if run[0] % show_step == 0:
                if loss_plots:
                    liveloss.draw()
                if show_synth:
                    plot_tensor_image(ax_synth, synth, 'synthesized');
                    display.display(fig)
                    display.clear_output(wait=True)
                
        self.base_synth = synth
        return synth
