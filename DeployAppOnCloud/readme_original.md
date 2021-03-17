[youtube](https://www.youtube.com/watch?v=rRMaQXIfngM&list=PLlH6o4fAIji6FEsjFeo7gRgiwhPUkJ4ap)
- mac桌面打开docker, 直接用terminal里面就有了
- 拿到一个ubuntu img作为infra
    - docker pull ubuntu (可试试docker run -it ubuntu bash)
- Build:
    - pip3 freeze > requirements.txt (最小化dependency!!!)
    - docker build -t sp500predictoronazure:latest . (在有Dockerfile的目录下)
- local跑:
    - docker run -p 5000:5000 sp500predictoronazure:latest
    - 在这里看：http://0.0.0.0:5000/
- push: 
    - docker tag sp500predictoronazure sxu11/dockerazure:1.0 (放到https://hub.docker.com/)
    - docker push sxu11/dockerazure:1.0 (先要login in)

my website: https://mloncloud.azurewebsites.net/

Azure: 
- management: portal.azure.com
    - App Services
    - Create (Web App)
    - Image and tag: sxu11/dockerazure:1.0
    - After complete: go to resource, 
        - page: https://sp500predictor.azurewebsites.net/
        - (Application) setting: add name=PORT, value=5000



