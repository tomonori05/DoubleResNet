from urllib import request
import os.path as ospath
import os
from keras.preprocessing.image import load_img, img_to_array

def HTTP_GET_ToText(url):
    with request.urlopen(url) as response:
        html = response.read().decode()
        return html

def DownloadWnid(wnid,save_folder,max=0):
    API_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={}"
    url = API_URL.format(wnid)
    image_list  = []

    html = HTTP_GET_ToText(url).split()
    file_name=html[::2]
    img_url=html[1::2]

    for i,f_name in enumerate(file_name):
        data={
            "filename":f_name,
            "img_url": img_url[i]
        }
        image_list.append(data)
    
    if not ospath.isdir(save_folder):
        os.makedirs(save_folder)

    downloadcount = 0
    for image in image_list:

        if max != 0 and max < downloadcount:
            break
        try:
            with request.urlopen(image["img_url"]) as response:
                img_res = response.read()
                save_path = save_folder+'/'+image["filename"]

                with open(save_path,'wb') as f:
                    f.write(img_res)
            
            downloadcount = downloadcount + 1
        except:
            print("error")

def Download_datasets(wnids,save_folder,max=0):
    SYNSET_GET_URL = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={}"
    
    if ospath.isdir(save_folder):
        os.makedirs(save_folder)

    for wid in wnids:
        synset_name = HTTP_GET_ToText(SYNSET_GET_URL.format(wid)).split()
        wid_folder = save_folder+'/'+synset_name[0]
        os.makedirs(wid_folder)
        DownloadWnid(wnid=wid,save_folder=wid_folder,max=max)

def MakeDataset(dataset_folder):
    files_dir = [f for f in os.listdir(path=dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    dataset = []

    for fdir in files_dir:
        folder_path=dataset_folder+'/'+fdir
        img_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(dataset_folder, f))]
        
        for img in img_files:
            img_path = dataset_folder+'/'+fdir+'/'+img
            img_array = img_to_array(load_img(img_path))
            data = {
                "img":img_array,
                "label":fdir
            }
            dataset.append(data)
    return dataset

