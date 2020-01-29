#from ..ResNetTester_cls import ResNetTester
import sys
sys.path.append('..')

#from ResNetTester_cls import ResnetTester
import json
import os

def json_write(tester,path):
    option = {
        "relu_option": str(tester.option["relu_option"]),
        "double_input": str(tester.option["double_input"]),
        "block": str(tester.option["block"]),
        "concatenate": str(tester.option["concatenate"]),
        "reseption" : str(tester.option["reseption"]),
        "dropout": str(tester.option["dropout"]),
        "filters": str(tester.option["filters"]),
        "wide": str(tester.option["wide"])
    }

    accuracys = [str(n) for n in tester.history.history['acc']]
    loss = [str(n) for n in tester.history.history['loss']]

    result_list = {
        "batch_size": str(tester.batch_size),
        "dataset": str(tester.dataset_name),
        "epoch": str(tester.nb_epoch),
        "vaildation_split": str(tester.validation_split),
        "test_datas": str(tester.test_data),
        "train_datas": str(tester.train_data),
        "accuracy": str(tester.accuracy),
        #"loss": str(tester.loss),
        "start_time": str(tester.start_time),
        "end_time": str(tester.end_time),
        "option": option,
        "acc": accuracys,
        "loss": loss
    }

    if os.path.exists(path):
        with open(path) as f:
            s = f.read()
            json_data = json.loads(s)
            json_data["result"].append(result_list)
    else:
        json_data = {
            "result": [result_list]
        }

    direcotry_separate = path.split('/')
    directory_name = ''

    for i in range(len(direcotry_separate)-1):
        directory_name += (direcotry_separate[i]+'/')
    
    if os.path.isdir(directory_name) == False:
        os.makedirs(directory_name)

    with open(path,mode='w') as f:
        #f.write(json_dump)
        json.dump(json_data,f,indent=4, sort_keys=True, separators=(',',': '))
