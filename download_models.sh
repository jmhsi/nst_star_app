#!/bin/sh

mkdir deployment/models && cd deployment/models # makes deployment/models and cds into it
wget -c --no-check-certificate https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth
cd ../.. #bring you back to project dir
