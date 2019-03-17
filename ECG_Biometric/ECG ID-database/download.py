
import urllib, urllib2, re, os

#url = 'https://physionet.org/physiobank/database/ecgiddb/Person_01/rec_1.hea'
#f = urllib.urlretrieve(url, 'rec1.txt')

url_ecgiddb = 'https://physionet.org/physiobank/database/ecgiddb/'
req = urllib2.Request(url_ecgiddb)
content = urllib2.urlopen(req).read()
num_persons = 90
for i in range(79,num_persons+1):
    if i < 10:
        curr_numStr = '0' + str(i)
    else:
        curr_numStr = str(i)

    curr_directory = 'Person_'+curr_numStr
    if not os.path.exists(curr_directory):
        os.makedirs(curr_directory)

    curr_url = url_ecgiddb+'Person_'+curr_numStr+'/'
    curr_req = urllib2.Request(curr_url)
    curr_content = urllib2.urlopen(curr_req).read()
    match_filename = re.compile(r'(?<=href=")(rec.*?)(?=")')
    filenames = re.findall(match_filename, curr_content)
    for filename in filenames:
        urllib.urlretrieve(curr_url+filename, curr_directory+'/'+filename)
