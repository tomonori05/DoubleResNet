import matplotlib.pyplot as plt
from datetime import datetime

def plot_acc_history(history, legend=['acc'], path='acc_history.png'):
    # print(history.history.keys())
    plt.clf()
    # 精度の履歴をプロット
    for h in history:
        plt.plot(h.history['acc'])
        
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(legend, loc='lower right')
    #plt.show()
    plt.savefig(path)

def plot_loss_history(history, legend=['loss'],path='loss_history.png'):
    plt.clf()
    # 損失の履歴をプロット
    for h in history:
        plt.plot(h.history['loss'])

    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(legend, loc='lower right')
    #plt.show()
    plt.savefig(path)

def outputfile_evalute(score,name,path='result.txt'):
    #結果のファイル出力
    with open(path,mode='a') as f:
        f.write('===' + name + '===\n')
        f.write('Test loss:' + str(score[0]) + '\n')
        f.write('Test accuracy:' + str(score[1])+'\n')

def output_text(text,path):
    with open(path,mode='a') as f:
        f.write('===='+ text + '====\n')

def get_time():
    #現在時刻を取得
    return datetime.now().strftime("%Y%m%d_%H%M%S")