#!/bin/sh

mkdir models && cd models # makes deployment/models and cds into it
wget -c --no-check-certificate https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth
cd .. # go back to deployment
