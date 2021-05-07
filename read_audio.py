# %%
import librosa
import numpy as np
import json


def read_wave(filepath: str,id: int):
    # 读取音频文件
    samplingRate = 16000
    x, sr = librosa.load(filepath, sr=samplingRate)
    # 去归一化
    x = x*32767
    # 写入到字典中去
    dic = dict()
    dic["000"] = dict()
    # 开始采样 10s采一次样
    for iteration, index in enumerate(range(0, len(x), 10*samplingRate)):
        tmp = dict()
        # 取音频切片
        # 不能变成10秒钟的采样的话，后面就取空
        if index+10*samplingRate > len(x):
            container = np.zeros((10 * samplingRate))
            container[0:len(x)-iteration *
                      samplingRate * 10] = x[iteration * samplingRate * 10:]
            tmp["audio_sequence"] = container.tolist()
        else:
            tmp["audio_sequence"] = x[index:index+10*samplingRate].tolist()
        # 骨骼取空数组
        tmp["joint_coors"] = np.zeros((100, 18, 2)).tolist()
        dic[str(id).zfill(3)][str(iteration*10).zfill(3)] = tmp

    return dic


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
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_to_json(dic, target_dir):
    dumped = json.dumps(dic, cls=NumpyEncoder)
    file = open(target_dir, 'w')
    json.dump(dumped, file)
    file.close()


# %%
if __name__ == "__main__":
    dic = read_wave("c:/users/ssk/desktop/test.mp3")
    save_to_json(dic, "c:/users/ssk/desktop/a.json")
