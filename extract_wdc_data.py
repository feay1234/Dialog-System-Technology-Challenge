import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from langdetect import detect
import gzip, argparse, glob
from collections import defaultdict


def getTitle(url):
    r = requests.get(url)
    soup = bs(r.content, 'lxml')
    return soup.select_one('title').text


def cleanhtml(raw_html):
    cleantext = bs(raw_html, "lxml").text
    cleantext = cleantext.replace("\n", " ").replace("\t", "")
    return cleantext


parser = argparse.ArgumentParser('Common Web Data Preprocessing')
parser.add_argument('--dir', type=str, default="/Users/jarana/Downloads/")
parser.add_argument('--nrows', type=int, default=100)
args = parser.parse_args()

files = []
for file in glob.glob("%s/*.gz" % args.dir):
    files.append(file)

for file in files:
    with open(file, 'rb') as fd:
        gzip_fd = gzip.GzipFile(fileobj=fd)
        df = pd.read_csv(gzip_fd, nrows=args.nrows, names=['text'], error_bad_lines=False)

    entityType = file.split("/")[-1].replace(".gz", "").replace("schema_", "")
    memo = defaultdict(dict)
    for idx, row in df.iterrows():

        l = row.text.split()
        try:
            if "schema.org" in l[1]:
                website = l[-2].replace("<", "").replace(">", "")
                # if "https://www" not in website:
                #     continue
                # p = l[1].split("/")[-1].replace(">","")
                # memo[website].add(p)
                p = l[1].split("/")[-1].replace(">", "")
                val = cleanhtml(row.text.split("\"")[1].encode('ascii', 'ignore').decode('unicode_escape'))
                # if detect(val) != "en":
                #     continue
                # print(l)
                # print(p)
                print(val)
                if p not in memo[website]:
                    memo[website][p] = val
        except:
            continue

    for website in memo:
        try:
            # if "description" in memo[website]:
            #     title = memo[website]['description'].encode('ascii', 'ignore').decode('unicode_escape')
            #
            #     if title == "":
            #         continue
            #
            #     title +=  + " " + entityType

                # with open(file.replace(".gz", ".tsv"), 'a') as the_file:
                #     for p in memo[website]:
                #         the_file.write('%s\t%s\n' % (title, p))
            # Get title from website
            # title = getTitle(website)
            # if detect(title) == "en":
            with open(file.replace(".gz", ".tsv"), 'a') as the_file:
                for p in memo[website]:
                    the_file.write('%s\t%s\t%s\t%s\n' % (entityType, website, p, memo[website][p]))
        except:
            continue
