How to run the docker and deploy the project

1. install docker

   ```shell
   pip install -y docker
   ```

1. build the images

   ```shell
   docker build -f dockerfile.txt -t b2b:1.0 .
   ```

1. run the container

   ```
   docker run -it -p 5000:5000 --name b2b b2b:1.0 /bin/bash
   ```

2. create or reconnect a terminal

   ```
   docker exec -it [container id] /bin/bash
   ```
   
5. deploy the project

   open the flaskproject folder and run the command:

   ```
   python ./App.py
   ```

6. stop the container

   ```
   docker stop [container id]
   ```

7. delete the container

   ```
   docker rm [container id]
   ```

8. delete the image

   ```
   docker rmi [container id]
   ```

   

