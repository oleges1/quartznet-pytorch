# never tested
FROM kaggle/python

# WORKDIR /home/user

# Requirements
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

CMD git clone --recursive https://github.com/parlance/ctcdecode.git
CMD cd ctcdecode && pip install .
CMD cd .. && rm -r ctcdecode

CMD python train.py --config=configs/train_LJSpeech.yaml
