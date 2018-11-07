from __future__ import print_function

import os
import sys
import torch
import io
import time
import numpy as np
from PIL import Image
import gc
import base64
import torch.onnx
from datetime import datetime
from torch.autograd import Variable
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET, set_config
from miscc.bird_config import bird_cfg
from miscc.coco_config import coco_cfg

from flask import Flask, request, Response, jsonify, send_file

from threading import Thread
import logging
from logging.handlers import RotatingFileHandler

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from werkzeug.contrib.cache import SimpleCache
#cache = SimpleCache()
cache = {}

app = Flask(__name__)
cfg = bird_cfg


# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')  # Put any other methods you need here

    return response

def vectorize_caption(wordtoix, caption, copies=2):
    # create caption vector
    tokens = caption.split(' ')
    cap_v = []
    for t in tokens:
        t = t.strip().encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i,:] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)

    #print(captions.astype(int), cap_lens.astype(int))
    #captions, cap_lens = np.array([cap_v, cap_v]), np.array([len(cap_v), len(cap_v)])
    #print(captions, cap_lens)
    #return captions, cap_lens

    return captions.astype(int), cap_lens.astype(int)

def generate_(caption, modelname, copies=2):
    # check cache contents
    if (modelname + '_wordtoix' in cache):
        print("cache contains:" + modelname + '_wordtoix')
    if (modelname + '_text_encoder' in cache):
        print("cache contains:" + modelname + '_text_encoder')

    wordtoix = cache[modelname + '_wordtoix']
    if (wordtoix == None):
        return 'invalid cache element'
    
    print('length(wordtoix): {}'.format(len(wordtoix)))

    text_encoder = cache[modelname + '_text_encoder']
    netG = cache[modelname + '_netG']
    # load word vector
    captions, cap_lens  = vectorize_caption(wordtoix, caption, copies)
    n_words = len(wordtoix)
    if (cap_lens[0] == 0):
        return "bad caption"
    # only one to generate
    batch_size = captions.shape[0]

    nz = cfg.GAN.Z_DIM
    captions = Variable(torch.from_numpy(captions), requires_grad = False).type(torch.LongTensor)
    cap_lens = Variable(torch.from_numpy(cap_lens), requires_grad = False).type(torch.LongTensor)
    noise = Variable(torch.FloatTensor(batch_size, nz), requires_grad = False)

    if cfg.CUDA:
        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        noise = noise.cuda()

    #######################################################
    # (1) Extract text embeddings
    #######################################################
    hidden = text_encoder.init_hidden(batch_size)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    mask = (captions == 0)
        

    #######################################################
    # (2) Generate fake images
    #######################################################
    noise.data.normal_(0, 1)
    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

    # ONNX EXPORT
    #export = os.environ["EXPORT_MODEL"].lower() == 'true'
    if False:
        print("saving text_encoder.onnx")
        text_encoder_out = torch.onnx._export(text_encoder, (captions, cap_lens, hidden), "text_encoder.onnx", export_params=True)
        print("uploading text_encoder.onnx")
        print("done")

        print("saving netg.onnx")
        netg_out = torch.onnx._export(netG, (noise, sent_emb, words_embs, mask), "netg.onnx", export_params=True)
        print("uploading netg.onnx")
        print("done")
        return

    # G attention
    cap_lens_np = cap_lens.cpu().data.numpy()


    # only look at first one
    #j = 0
    #for j in range(batch_size):
    #    for k in range(len(fake_imgs)):

    im = fake_imgs[2][0].data.cpu().numpy()
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)

    # save image to stream
    stream = io.BytesIO()
    im.save(stream, format="png")
    im_256 = base64.b64encode(stream.getvalue()).decode()

    im = fake_imgs[1][0].data.cpu().numpy()
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)

    # save image to stream
    stream = io.BytesIO()
    im.save(stream, format="png")
    im_128 = base64.b64encode(stream.getvalue()).decode()

    im = fake_imgs[0][0].data.cpu().numpy()
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)

    # save image to stream
    stream = io.BytesIO()
    im.save(stream, format="png")
    im_64 = base64.b64encode(stream.getvalue()).decode()

    ret = [{'im_256': im_256, 'im_128': im_128, 'im_64': im_64}]
    return ret


def word_index(model):
    # model must be 'bird' or 'coco'
    ixtoword = cache.get(model + '_ixtoword', None)
    wordtoix = cache.get(model + '_wordtoix', None)
    if ixtoword is None or wordtoix is False:
        #print("ix and word not cached")
        # load word to index dictionary
        x = pickle.load(open('/home/bitnami/apps/AttnGAN/server' + '/data/' + model + '/captions.pickle', 'rb'))
        ixtoword = x[2]
        wordtoix = x[3]
        print('length(wordtoix): {}'.format(len(wordtoix)))
        cache[model + '_ixtoword'] = ixtoword
        cache[model + '_wordtoix'] = wordtoix

    return wordtoix, ixtoword

def models(modelname, cfg, word_len):
    #print(word_len)
    text_encoder = cache.get(modelname + '_text_encoder', None)
    if text_encoder is None:
        #print("text_encoder not cached")
        text_encoder = RNN_ENCODER(word_len, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        if cfg.CUDA:
            text_encoder.cuda()
        text_encoder.eval()
        cache[modelname + '_text_encoder'] = text_encoder


    netG = cache.get(modelname + '_netG', None)
    if netG is None:
        #print("netG not cached")
        netG = G_NET()
        state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        if cfg.CUDA:
            netG.cuda()
        netG.eval()
        cache[modelname + '_netG'] = netG

    return text_encoder, netG

def eval(caption):
    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))

    t0 = time.time()
    urls = generate(caption, wordtoix, ixtoword, text_encoder, netG)
    t1 = time.time()

    response = {
        'small': urls[0],
        'medium': urls[1],
        'large': urls[2],
        'map1': urls[3],
        'map2': urls[4],
        'caption': caption,
        'elapsed': t1 - t0
    }

    return response

@app.route('/init/<modelname>')
def init(modelname):
    if (os.name == 'posix'):
        handler = RotatingFileHandler('/home/bitnami/apps/AttnGAN/mesg.log', maxBytes=10000, backupCount=1)
    if (os.name == 'nt'):
        handler = RotatingFileHandler('C:\Telesens\web\AttnGAN\client\mesg.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)
    try:
        if modelname == 'bird':
            cfg = bird_cfg
            set_config(cfg)
        if modelname == 'coco':
            cfg = coco_cfg
            set_config(cfg)
        if (os.name == 'posix'):
            cfg.CUDA = False
        if (os.name == 'nt'):
            cfg.CUDA = True

        wordtoix, ixtoword = word_index(modelname)
        # lead models
        text_encoder, netG = models(modelname, cfg, len(wordtoix))
   #     app.logger.info('successfully initialized AttnGAN')
        return Response('successfully initialized AttnGAN')
    except Exception as e:
   #     app.logger.info('AttnGAN initialization error: %e' % e)
        return Response(e)


@app.route('/generate/<caption>/<modelname>', methods=['POST'])
def generate(caption, modelname):
    try:
        print("In generate: args1: {}, arg2: {}".format(caption, modelname))
        gc.collect()        
        t0 = time.time()
        stream = generate_(caption, modelname)
        t1 = time.time()
        print(t1 - t0)
        stream[0]['fp_time'] = t1-t0
        
        return jsonify(stream)
    except Exception as e:
    #    app.logger.info('Error while generating image: %e' % e)
        print(e)
        return Response(e)

if __name__ == "__main__":
    if (os.name == 'nt'):
        # without SSL
        app.run(debug=True, host='0.0.0.0', port=5000)

    if (os.name == 'posix'):
        # without SSL
        app.run(debug=True, host='0.0.0.0')

        #app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))

    #caption = "the bird has a yellow crown and a black eyering that is round"


