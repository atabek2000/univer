import numpy as np
import math as m
# класс для tfidf
class TfIdf():
  # конструктор класса
  def __init__(self, text_arr):
    self.text_arr = text_arr

  # объединяем вес текст
  def join_doc_words(self):
    words = ' '.join(self.text_arr)
    return words

  # получим уникальные значение
  def get_unique(self):
    return sorted(set(self.join_doc_words().split(' ')))

  # вычислим tf
  def tf(self,t,d):
    return d.count(t)

  # вычислим df
  def df(self, t):
    cn = 0
    for i in self.text_arr:
      if i.count(t) > 0:
        cn += 1
    return cn

  # вычислим idf
  def idf(self, t):
    return np.log((1 + len(self.text_arr))/(self.df(t) + 1)) + 1

  # вычислими не нормализованный tfidf
  def tfidf_(self):
    tfidf = list()
    for doc in self.text_arr:
      elem = list()
      for word in self.get_unique():
        tf_ = self.tf(word, doc)
        idf_ = self.idf(word)
        elem.append(tf_*idf_)
      tfidf.append(elem)
    return tfidf

  # нормализуем tfidf
  def to_array(self):
    tfidf_arr = self.tfidf_()
    w_ = len(tfidf_arr)
    h_ = len(tfidf_arr[0])
    norm = np.full((w_, h_), 0.0)
    for i in range(w_):
      for j in range(h_):
        norm[i][j] = tfidf_arr[i][j]/m.sqrt(sum([x**2 for x in tfidf_arr[i]]))
    return norm

  # уникальные слова в тексте
  def get_feature_names(self):
    return self.get_unique()


texts = ['the man went out for walk',
         'the children sat around the fire',
         'the man walk alone'
         ]

tfidf = TfIdf(texts)
print(tfidf.get_feature_names())
tfidf.to_array()
