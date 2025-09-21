import json

with open('/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/train_data/t2i_train_data_720_exists.json') as f:
    data=json.load(f)
sample_data=data[:10]
for item in sample_data:
    item['image_url']=item['image']
    item['text']='please descript the image'

with open('sample_data.json','w') as f:
    json.dump(sample_data,f,indent=4)