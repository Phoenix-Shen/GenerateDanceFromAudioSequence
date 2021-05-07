# %%
import librosa
import numpy as np
import json
import os 

####################json文件编码######################
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj) 

#写进JSON文件
def save_to_json(dic,target_dir):
    dumped = json.dumps(dic, cls=NumpyEncoder)  
    file = open(target_dir, 'w')  
    json.dump(dumped, file)
    file.close()


#读取JSON文件
def read_from_json(target_dir):
    f = open(target_dir,'r')
    data = json.load(f)
    #需要dump一下
    data = json.dumps(data)
    data = json.loads(data)
    f.close()
    return data 

#########################################################
#读取一个json文件的坐标，并变成(18,2)维度
def read_json_coors(coor_file_path:str):
    data =read_from_json(coor_file_path)
    person_data=json.dumps(data['people'][0])
    person_data=json.loads(person_data)
    coor_data=person_data['pose_keypoints_2d']
    coors=np.zeros((18,2))
    for i in range(0,len(coor_data),3):
        #print(i/3)
        coors[int(i/3)][0]=coor_data[i]
        coors[int(i/3)][1]=coor_data[i+1]
    return coors

def get_one_sequence(coors_folder_path:str,audio_path:str):

    #获取JSON文件的目录
    json_coors= os.listdir(coors_folder_path)
    json_coors=sorted(json_coors)
    os.chdir(path)

    # 读取音频文件
    samplingRate = 16000
    x, sr = librosa.load(audio_path, sr=samplingRate)
    # 去归一化
    x = x*32767

    dic=dict()
    # 开始采样 10s采一次样
    for iteration, index in enumerate(range(0, len(x), 10*samplingRate)):
        tmp =dict()
        # 处理音频
        # 取音频切片
        # 不能变成10秒钟的采样的话，后面就取空
        if index+10*samplingRate > len(x):
            container = np.zeros((10 * samplingRate))
            container[0:len(x)-iteration *
                      samplingRate * 10] = x[iteration * samplingRate * 10:]
            tmp["audio_sequence"] = container.tolist()
        else:
            tmp["audio_sequence"] = x[index:index+10*samplingRate].tolist()
        # 骨骼处理
        joint_coors_data=np.zeros((100, 18, 2)).tolist()
        # 一个采样需要100个骨骼数据
        for iteration2,index2 in enumerate(range(iteration*100,iteration*100+100)):
            joint_coors_data[iteration2]=read_json_coors(json_coors[index2])
        tmp["joint_coors"]=joint_coors_data
        #时间数据
        dic[str(id).zfill(3)][str(iteration*10).zfill(3)] = tmp

    return dic



#%%
if __name__=="__main__":
    pass
# %%
