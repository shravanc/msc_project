import csv
import json
import os
from utils import get_text, get_polarity



file_name = '../covid_data.jsonl'
original_file = '/home/shravan/Downloads/aylien-covid-news.jsonl'
base_csv_name = "./csv_files"

def read_file(file_name):
  split_duration = 400 #1000

  file_pointer = None
  with open(file_name) as fp:
    lines = fp.readlines()
    for i, line in enumerate(lines):
      print("i-------->", i)

      if (i%split_duration)==0:
        print("i*******>", i)
        if file_pointer is not None:
          file_pointer.close()
        fname = os.path.join(base_csv_name, f"text_{i}.csv")
        file_pointer = open(fname, mode="w")
        fp = csv.writer(file_pointer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)


      j_data = json.loads(line)
      try:
        text = get_text(j_data)
        polarity = get_polarity(j_data)
        if len(text) == 0:
          continue
      except:
        continue

      fp.writerow([text, polarity])





#read_file(file_name)
read_file(original_file)
