#%%
import librosa
import numpy as np

def read_wave(filepath:str):
    #读取音频文件
    samplingRate=16000
    x,sr =librosa.load(filepath,sr=samplingRate)
    #去归一化
    x=x*4294967295
    #写入到字典中去
    dic =dict()
    dic["000"]=dict()
    #开始采样
    for iteration,index in enumerate(range(0,len(x),10*samplingRate)):
        tmp=dict()
        #取音频切片
        #不能变成10秒钟的采样的话，后面就取空
        if index+10*samplingRate>len(x):
            tmp["audio_sequence"]=np.zeros(10*samplingRate)
            tmp["audio_sequence"][0:len(x)-iteration*samplingRate]=x[index:]
        else:
            tmp["audio_sequence"]=x[index:index+10*samplingRate]
        #骨骼取空数组
        tmp["joint_coors"]=np.zeros((50,18,2))
        dic["000"][str(iteration)]=tmp
    
    return tmp
    

#%%
if __name__ =="__main__":
    read_wave("c:/users/ssk/desktop/test.mp3")