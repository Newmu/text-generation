from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import numpy as np
import cPickle
from datetime import datetime
from time import time,sleep
from sklearn.cross_validation import train_test_split
import sklearn.linear_model as lm
from sklearn import metrics

texts = []
for i in range(1000):
	print i
	link = 'http://en.wikipedia.org/wiki/Special:Random'
	headers = {}
	headers['User-Agent'] = "Mozilla/5.0 (X11; Linux x86_64)"
	r = requests.get(link, headers=headers)
	soup = BeautifulSoup(r.text, "lxml")
	text_wrap = soup.find('div',{"id" : "mw-content-text"})
	text_tags = text_wrap.find_all('p')
	text = ' '.join([p.get_text().strip().encode('ascii', 'ignore') for p in text_tags])
	texts.append(text)
	sleep(1)

print len(' '.join(texts))
cPickle.dump(texts,open('data/data.pkl','wb'))