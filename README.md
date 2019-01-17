### Setting up a NST app, with style loss using L2 Wasserstein Distance

Try it on google cloud platform: http://35.192.6.67

## To download model weights
```
cd deployment
sh download_models.sh
```

## To run locally
```
cd deployment
python nst_star_app.py serve
```
Go to localhost:8008

## To run in a docker container
```
docker image build -t nst_star_app:latest .
docker run -d --name nst_star_app --publish 8008:8008 nst_star_app:latest 
```
Go to localhost:8008. you can change --publish \<choose port>:8008 and could go to localhost:\<choose port> instead

## TODO
1) Add selection for number of steps (currently having a problem of FormData corruption when including form.append('steps', steps) in client.js)
2) Display stylized image after every 5 steps and/or display an animated gif showing the style change after every n steps. 1) must be fixed first.
3) Charbonnier loss for content?

## Papers and resources
https://arxiv.org/pdf/1808.03344.pdf (NST literature overview)
https://arxiv.org/pdf/1705.04058.pdf (Another ST overview)
https://arxiv.org/pdf/1807.05927.pdf (Charbonnier Loss)
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf (gatys paper 1?)
https://arxiv.org/pdf/1611.07865.pdf (Gatys paper 2)
http://cs231n.stanford.edu/reports/2017/pdfs/402.pdf (depth perception with NST)
https://arxiv.org/pdf/1610.07629.pdf (Neural NST methods and multiple styles)
https://towardsdatascience.com/practical-techniques-for-getting-style-transfer-to-work-19884a0d69eb (helpful NST info)
https://distill.pub/2018/differentiable-parameterizations/ (3D style transfer)
https://github.com/render-examples/fastai-v3 (deployment)
https://course-v3.fast.ai/deployment_zeit.html (deployment)
https://github.com/nikhilno1/healthy-or-not/blob/master/heroku-deploy.md (deployment)
https://github.com/VinceMarron/style_transfer (Why L2 Wasserstein loss)


## Some examples
Golden Gate and cityscape styled as a tree and cartoon bears
![Alt text](/images/ex1.png?raw=true "Optional Title")
![Alt text](/images/ex2.png?raw=true "Optional Title")
