import json
import csv

csv_fname = './csv_files/covid_data_'
#fname = '/home/shravan/Downloads/aylien-covid-news.jsonl'
fname = '../covid_data.jsonl'
fp = open(fname)

total = 528847

with open('employee_file.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    employee_writer.writerow(['John Smith', 'Accounting', 'November'])
    employee_writer.writerow(['Erica Meyers', 'IT', 'March'])


def write_to_file(name, text, label):
    json_data = json.dumps(data)
    with open(name, mode='w') as fp:
      f = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      f.writerow([text, label])
      




    f.write(json_data)
    f.close()


def construct_string(sentences, sentiment):
    text = ""
    # try:
    polarity = sentiment['body']['polarity']
    if polarity == 'positive':
        polarity = 1
    else:
       polarity = 0
    # except:
    #     print("---except---")
    #     return ""

    for sentence in sentences:
        text += sentence.replace(',', '')

    return text, polarity


split_time = 50000
data = ""
error_count = 0
error_index = []
for i, line in enumerate(fp.readlines()):
    if i == 3:
        break

    if (i % split_time) == 0:
        fname = csv_fname + str(i) + '.csv'
        write_to_file(fname, data)
        data = ""

    js = json.loads(line)
    try:
        data += construct_string(js['summary']['sentences'], js['sentiment'])
        print("****************", i)
    except:
        print('******----*******')
        error_count += 1
        error_index.append(i)

fp.close()

fname = csv_fname + 'final.csv'
write_to_file(fname, data)
#
# dumps = json.dumps(datastore)
# print(dumps)
# fp = open(process_json, 'w+')
# fp.write(dumps)
