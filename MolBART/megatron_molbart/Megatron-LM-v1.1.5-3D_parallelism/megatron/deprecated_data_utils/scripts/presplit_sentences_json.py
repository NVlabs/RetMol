# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
python scripts/presplit_sentences_json.py <original loose json file> <output loose json file>
"""

import sys
import json

import nltk

nltk.download('punkt')

input_file = sys.argv[1]
output_file = sys.argv[2]

line_seperator = "\n"

with open(input_file, 'r') as ifile:
    with open(output_file, "w") as ofile:
        for doc in ifile.readlines():
            parsed = json.loads(doc)
            sent_list = []
            for line in parsed['text'].split('\n'):
                if line != '\n':
                    sent_list.extend(nltk.tokenize.sent_tokenize(line))
            parsed['text'] = line_seperator.join(sent_list)
            ofile.write(json.dumps(parsed) + '\n')
