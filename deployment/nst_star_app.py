
import vgg_classes as vg
import torch
import matplotlib.pyplot as plt

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from starlette.staticfiles import StaticFiles
#from wtforms import (Form, FileField, IntegerField, SubmitField, validators)

from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio
import os
import json
import requests
import base64 
from PIL import Image as PILImage
import torchvision.transforms as transforms

app = Starlette(debug=True, template_directory='templates')
cur_dir = os.getcwd()
app.mount('/statics', StaticFiles(directory='statics'), name='statics')

# setup
# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# choose layers
cont_lyrs = ['relu1_1']
stl_lyrs = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1'] 
# NN choices
pool_type = 'avg'
cont_loss_type = 'mse'
stl_loss_type = 'l2'

# load in gatys vgg, use as base for our vgg
model_dir = os.getcwd() + '/models/'
vgg_g = vg.VGG_g(pool='avg')
vgg_g.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))


@app.route("/", methods=["GET"])
async def index(request):
    template = app.get_template('index_form.html')
    content = template.render(request=request)
    return HTMLResponse(content)

@app.route("/stylize", methods=["POST"])
async def stylize(request):
    data = await request.form()
    content = await (data["content"].read())
    style = await (data["style"].read())
    steps = 30
    decoded_bytes_img, h, w = do_nst_stylize(content, style, steps)
    print('sending bytes img in jsonresponse')
    return JSONResponse({'stylized_image': decoded_bytes_img,
                         'h': h,
                         'w': w,
                         'steps': steps})
    
def encode(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = vg.unloader(image)
    image[image>1] = 1
    image[image<0] = 0
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(image)
#     return pil_image
    buff = BytesIO()
    pil_image.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")    

def do_nst_stylize(content, style, steps=10):
    content_img = vg.image_loader(BytesIO(content))
    style_img = vg.image_loader(BytesIO(style))
    synth = content_img.clone()
    bs, c, h,w = content_img.shape
    nst = vg.TransferStyle(base_cnn=vgg_g.to(device),
                           cont_loss_type=cont_loss_type,
                           stl_loss_type=stl_loss_type,
                           cont_img=content_img,
                           stl_img=style_img,
                           pool_type=pool_type,
                           cont_lyrs=cont_lyrs,
                           stl_lyrs=stl_lyrs,
                           device=device,
                           resize_conv=False,
                           downsample=False,
                           approx_stl=True)
    output = nst.run_style_transfer(synth, 'adam', steps, 1e0, style_off=False,
                                    content_off=False, show_step=int(steps/5),
                                    loss_plots=False, show_synth=False)
    o_img_data = encode(output)
    return o_img_data, h, w

if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host="0.0.0.0", port=port)