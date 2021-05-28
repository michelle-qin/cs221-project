# -*- coding: utf-8 -*-
"""Convert the Yelp Dataset Challenge dataset from json format to csv.

For more information on the Yelp Dataset Challenge please visit http://yelp.com/dataset_challenge

"""
import argparse
import collections
import csv
import simplejson as json
import re


def read_and_write_file(json_file_path, csv_file_path, column_names):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    with open(csv_file_path, 'w') as fout:
        csv_file = csv.writer(fout)
        #print(column_names)
        csv_file.writerow(list(column_names))
        with open(json_file_path) as fin:
            count = 0
            #print(fin[0])
            for line in fin:
                if (line.find("\"date\":\"2020") == -1):
                    continue
               
                if (count > 10000):
                    #print("help")
                    break
                count += 1
                line_contents = json.loads(line)
                csv_file.writerow(get_row(line_contents, column_names))

def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""
    column_names = set()
    #print("1234567890")
    with open(json_file_path) as fin:
        count = 0
        for line in fin:
            count += 1
            if (count > 10000):
                break
            line_contents = json.loads(line)
            column_names.update(
                    set(get_column_names(line_contents).keys())
                    )
    #print("abcde")
    #print(column_names)
    return column_names

def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.

    Example:

        line_contents = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }

        will return: ['a.b', 'a.c']

    These will be the column names for the eventual csv file.

    """
    column_names = []
    for k, v in line_contents.items():
        if (k == "cool" or k == "useful" or k == "funny" or k == "review_id" or k == "business_id"):
            continue
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                    get_column_names(v, column_name).items()
                    )
        else:
            column_names.append((column_name, v))
    #print(column_names)
    return dict(column_names)

def get_nested_value(d, key):
    """Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.
    
    Example:

        d = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        key = 'a.b'

        will return: 2
    
    """
    if '.' not in key:
        if key not in d:
            return None
        return d[key]
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    return get_nested_value(sub_dict, sub_key)

def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    #print(column_names)
    for column_name in column_names:
        if (column_name == "cool" or column_name == "useful" or column_name == "funny" or column_name == "review_id" or column_name == "business_id"):
            continue
        #print(column_name)
        

        line_value = get_nested_value(
                        line_contents,
                        column_name,
                        )
        
        if (column_name == "stars"):
            if (line_value == 1.0 or line_value == 2.0 or line_value == 3.0):
                line_value = 0
            else:
                line_value = 1.0
        if (column_name == "text"):

            #print(line_value)
            
            line_value = re.sub(r'[^\w\s]', '', line_value)
            line_value = line_value.lower()
            #print(line_value)
        # if (column_name == "date"):
        #     if (line_value[0:4] != "2020"):
        #         continue

                #### WE ARE HERE

        if isinstance(line_value, str):
            
            row.append('{0}'.format(line_value)) #removed decoding
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
        #print("end of function?")
    return row

if __name__ == '__main__':
    """Convert a yelp dataset file from json to csv."""

    parser = argparse.ArgumentParser(
            description='Convert Yelp Dataset Challenge data from JSON format to CSV.',
            )

    parser.add_argument(
            'json_file',
            type=str,
            help='The json file to convert.',
            )

    args = parser.parse_args()

    json_file = args.json_file
    csv_file = "dataset_bert.csv"
    #csv_file = '{0}.csv'.format(json_file.split('.json')[0])

    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)
