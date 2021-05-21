import json

import csv

#yelp_data = open("yelp_academic_dataset_review.json")
with open("shortreviews.json") as f:
    data = json.loads(f.read())
    #print(data[0]['text'])
data = json.dumps(data)
yelp_parsed = json.loads(data) #yelp_data??


# open a file for writing

yelp_csv_data = open('/tmp/YelpData.csv', 'w')

# create the csv writer object

csvwriter = csv.writer(yelp_csv_data)
count = 0
with open('YelpData.csv', 'w', newline='') as file2:
    res_header = yelp_parsed[0]
    header = res_header.keys()
    header_trimmed = []
    for h in header:
        if (h == "stars" or h == "user_id" or h == "text"):
            header_trimmed.append(h)
    writer = csv.writer(file2)
    writer.writerow(header_trimmed)
    
    for res in yelp_parsed:
        res_values_trimmed = []
        #print(res)
        #print(res.keys())
        #csvwriter.writerow(header)
        #print(res.values())
        res_values_trimmed.append(res["user_id"])
        res_values_trimmed.append(res["stars"])
        res_values_trimmed.append(res["text"])
        writer.writerow(res_values_trimmed)
        #writer.writerow(res.values())
        count += 1
        #print(res.values())
        #csvwriter.writerow(res.values())
    yelp_csv_data.close()
    file2.close()
    f.close()