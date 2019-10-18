from __future__ import print_function
import json, math, time
import os.path, re, pinyin
from os.path import join, dirname, exists
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback, AudioSource
import threading, sys
from io import TextIOWrapper
sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import glob, itertools
import pickle
import difflib
import subprocess
from pydub import AudioSegment

# Last modified October 18, 2019 with comments
# From a language learning textbook that comes with MP3 recordings of example
# sentences used in the book, extract:
# 1. the MP3s and put them into the collection_media folder
# 2. the sentences to match to segments of the MP3s and put them into text_sentences
# Program works by:
# Running the IBM Watson speech2text service on each MP3 (which has the audio
#  for an entire chapter (many sentences))
# Formatting the Watson output into one big list of watson_3tuples (the predicted
#  word, start time, end time)
# For each text sentence, matching it to the concatenation of 3tuples of similar size
# (many words joined together to make up a sentence of similar length to the text)
#  which scores highest on the similarity metric 
# Cutting the mp3 at the start and end times of the concatenation
# Outputing a tab-delimited file mapping the text sentence to the mp3

out_file = open('c8000_watson.txt', 'w', encoding='utf-8')
collection_media = './collection.media/'

# Python pickle module allows you to cache the Watson output
if os.path.isfile('c8000.p'):
    picklecache = pickle.load(open('c8000.p', 'rb'))
else:
    picklecache = dict()
    pickle.dump(picklecache, open('c8000.p', 'wb'))

# this API key is disabled
service = SpeechToTextV1(
    url='https://stream.watsonplatform.net/speech-to-text/api',
    iam_apikey='kWY0QEEra5aA9kx_tyxF3Bnz3FXYwnW3Am7K7vZp4bRK')

mp3_path = './MP3/'
input_mp3s = glob.glob(mp3_path + '*.mp3')
input_mp3s = sorted(input_mp3s)
print(input_mp3s)

for i in range(len(input_mp3s)):
  input_mp3 = input_mp3s[i]

  input_mp3_filename = input_mp3.split('/')[-1]
  print(input_mp3_filename)
  sys.stdout.flush()

  if input_mp3_filename not in picklecache:
      with open(input_mp3,'rb') as audio_file:
          json_loads = service.recognize(
              model='zh-CN_BroadbandModel',
              audio=audio_file,
              content_type='audio/mp3',
              timestamps=True,
              speaker_labels=False,
              word_confidence=False,
              profanity_filter=False,
              smart_formatting=True).get_result()
          picklecache[input_mp3_filename] = json_loads
          print(json_loads)
          pickle.dump(picklecache, open('c8000.p', 'wb'))

  else:
      json_loads = picklecache[input_mp3_filename]

# Similarity metric: average of 
# similarity in terms of Chinese characters
# similarity in terms of pinyin romanization, which helps since Watson
#  will guess a homophone of the actual character, and the two will have
#  the same pinyin but not the same character
seq_hanzi = difflib.SequenceMatcher(None, 'x', 'y')
seq_pinyin = difflib.SequenceMatcher(None, 'x', 'y')

def ratio4sort(a, b):
  seq_hanzi.set_seq1(a)
  seq_hanzi.set_seq2(b)
  seq_pinyin.set_seq1( pinyin.get(a, format='strip', delimiter=' ') )
  seq_pinyin.set_seq2( pinyin.get(b, format='strip', delimiter=' ') )
  if seq_hanzi.real_quick_ratio() > 0.5 or seq_pinyin.real_quick_ratio() > 0.5:
    d1 = seq_hanzi.ratio()
    d2 = seq_pinyin.ratio()
    return 0.5*(d1+d2)
  else:
    return 0

# compile all watson_watson_3tuple
watson_watson_3tuple = list()
for x in picklecache.items():
  mp3 = x[0]
  print(mp3)
  json_loads = x[1]
  if 'results' in json_loads and len(json_loads['results']) > 0 and '00.mp3' not in mp3:
    for result in json_loads['results']:
      timestamps = result['alternatives'][0]['timestamps']
      watson_watson_3tuple += ([ x + [mp3] for x in timestamps ])
# a triplet is [hanzi, start s, end s, mp3]

text_sentences = open('./c8000_zh.txt', 'rU', encoding='utf-8').read().splitlines()
text_sentences = [ x.split('\t')[0] for x in text_sentences ]
text_sentences = [x for x in text_sentences if len(x) > 4 ]

if exists('ankikey2testsent_cache.p'):
  ankikey2testsent_cache = pickle.load(open('ankikey2testsent_cache.p', 'rb'))
else:
  ankikey2testsent_cache = dict()

# Search is difficult, since text sentences may be ordered differently from mp3 sentences
# Approach is somewhat optimized, basically start looking at concatenations of watson_3tuples
# around the index of the last match. Say the last text sentence matched to index 1234 of the
# list of 3tuples. Then look at 1233, then 1235, 1232, 1236, ...
prioritized_i = 0
window_radius = 5000
best_ratio = 0
best_i = 0
min_mp3 = 'A'
max_mp3 = 'Z'
running_mp3s = list(['M', 'M', 'M'])
total_duration = total_hanzi = 0

for anki_key_i in range(len(text_sentences)):
  anki_key = text_sentences[anki_key_i]
  anki_key_clean = trad2simp(re.sub('[^一-龯]', '', anki_key))
  print('mp3: %s(%s)' % (anki_key, anki_key_clean))

  if anki_key in ankikey2testsent_cache:
    print('\tCached.')
    best_test_sentence = ankikey2testsent_cache[anki_key]

    if best_test_sentence['ratio'] == 1.0 and len(best_test_sentence['sent']) > 7:
      min_mp3 = max(min_mp3, best_test_sentence['mp3'])
      print('\tmin_mp3=%s' % min_mp3)

  else:
    print('\tLooking up.')
    time.sleep(0.001)

    window_right = max(best_i + 5000, round(anki_key_i / len(anki_keys) * len(watson_watson_3tuple) * 2))
    window_right = min(window_right, len(watson_watson_3tuple))
    window = [x for x in watson_watson_3tuple[0:window_right]]

    anki_key_clean = trad2simp(re.sub('[^一-龯]', '', anki_key))
    test_len = len(anki_key_clean)

    l1 = range(1, len(window)-best_i-1)
    l2 = range(0, -best_i, -1)
    prioritized_window_indices = [val+best_i for pair in itertools.zip_longest(l1, l2) for val in pair if val]

    best_ratio = 0
    best_test_sentence = None

    while len(prioritized_window_indices):
      i = prioritized_window_indices.pop(0)
      watson_3tuple = list([window[i]])
      j = 1
      while sum( [ len(x[0]) for x in watson_3tuple ] ) < test_len and i+j < len(window):
        watson_3tuple.append( window[i+j] )
        j += 1

      test_sentence = {'sent': ''.join([ x[0] for x in watson_3tuple]), 'start': watson_3tuple[0][1], 'end': watson_3tuple[-1][2], 'mp3': watson_3tuple[0][3]}
      test_sentence_text = ''.join([ x[0] for x in watson_3tuple])

      if test_sentence['mp3'] < min_mp3:
        ratio = 0
      else:
        ratio = ratio4sort(anki_key_clean, test_sentence_text)

      if ratio > best_ratio:
        best_ratio = ratio
        best_test_sentence = test_sentence
        best_i = i
        print('\t%s with ratio %0.2f. Index was %d out of %d.' % (test_sentence, ratio, i, len(window)))
        if ratio > 0.75 and len(prioritized_window_indices) > 10:
          prioritized_window_indices = prioritized_window_indices[0:10]
          print('\t\tCutting pwi to len %d' % len(prioritized_window_indices))
      if ratio == 1.0:
        min_mp3 = test_sentence['mp3']
        print('\t\tRatio=1.0. min_mp3=%s. Break.' % min_mp3)
        break

    best_test_sentence['ratio'] = best_ratio
    ankikey2testsent_cache[anki_key] = best_test_sentence

    if anki_key_i % 12 == 0:
      pickle.dump(ankikey2testsent_cache, open('./ankikey2testsent_cache.p', 'wb'))

  best_ratio = best_test_sentence['ratio']
  start_time = best_test_sentence['start']
  end_time = best_test_sentence['end']
  mp3_to_split = best_test_sentence['mp3']
  mp3_to_split_w_path = mp3_path + mp3_to_split
  mp3_out = '%s_%d.mp3' % (mp3_to_split.replace('.mp3', ''), start_time*100)
  mp3_out_w_path = collection_media + mp3_out

  total_duration += (end_time - start_time)
  total_hanzi += len(anki_key_clean)

  if best_ratio > 0.5:
    if not exists(mp3_out_w_path):
      print('\t\tSplitting %s at (%s : %s)' % (mp3_to_split, start_time, end_time))
      time.sleep(0.1)

      # Useful python module
      song = AudioSegment.from_mp3( mp3_to_split_w_path )
      duration = song.duration_seconds
      sound_begin_s = max(0, start_time*1000 - 500)
      sound_end_s = min(duration*1000, end_time*1000 + 500)
      sound_split = song[sound_begin_s:sound_end_s]
      sound_split.export( mp3_out_w_path, format='mp3' )

      print('\t\tOutput: %s' % mp3_out_w_path)

    out_file.write(anki_key)
    out_file.write('\t')
    out_file.write('[sound:' + mp3_out + ']')
    out_file.write('\n')
out_file.close()
