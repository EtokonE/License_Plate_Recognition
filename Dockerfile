FROM pytorch/pytorch
USER root
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y python3-distutils

RUN useradd --no-user-group --create-home --shell /bin/bash lpr
USER lpr
WORKDIR /home/lpr

RUN pip3 install --upgrade pip
ENV PATH="/home/lpr/.local/bin:${PATH}"
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python3 -c "import cv2"
RUN python3 -c "from flask import Flask, jsonify, request"
USER root
RUN mkdir /home/lpr/License_Plate_Recognition
RUN chown -R lpr /home/lpr/License_Plate_Recognition
USER lpr
ADD . /home/lpr/License_Plate_Recognition
WORKDIR /home/lpr/License_Plate_Recognition
CMD [ "python3" , "/home/lpr/License_Plate_Recognition/app.py" ]
