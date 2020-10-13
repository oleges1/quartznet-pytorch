# never tested
FROM kaggle/python

# WORKDIR /home/user

# Requirements
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

CMD python train.py --config=configs/train_LJSpeech.yaml
