FROM tensorflow/tensorflow:1.13.1-gpu-py3

WORkDIR /project
COPY . .
RUN pip3 install -r requirements.txt

CMD ["./docker_run.sh", "--data_root", "/data/training"]
