### Setting up a NST app, with style loss using L2 Wasserstein Distance

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

## Some examples
Golden Gate and cityscape styled as a tree and cartoon bears
![Alt text](/images/ex1.png?raw=true "Optional Title")
![Alt text](/images/ex2.png?raw=true "Optional Title")
