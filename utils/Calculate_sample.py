# 计算txt标签格式每个类别的数量
import os
from tqdm import tqdm
def get_every_class_num(txt_path):
    # 根据自己的类别，注意一一对应
    class_categories=['BUS LANE', 'Jeltaya razmetka', 'Liniya 1', 'Liniya 2', 'Perehod', 'Romb', 'SLOW',
        'Strelka vlevo', 'Strelka vpered', 'Strelka vpered - vlevo', 'Strelka vpered - vpravo', 'Strelka vpravo', 'Velosiped']
    class_num = len(class_categories)  # 样本类别数
    class_list = [i for i in range(class_num)]
    class_num_list = [0 for i in range(class_num)]
    labels_list = os.listdir(txt_path)
    for i in tqdm(labels_list):
        file_path = os.path.join(txt_path, i)
        file = open(file_path, 'r')  # 打开文件
        file_data = file.readlines()  # 读取所有行
        for every_row in file_data:
            class_val = every_row.split(' ')[0]
            class_ind = class_list.index(int(class_val))
            class_num_list[class_ind] += 1
        file.close()
    # 输出每一类的数量以及总数
    result=dict(zip(class_categories,class_num_list))
    for name,num in result.items():
        print(name,":",num)
    print("-----------------------------------")
    print('total:', sum(class_num_list))
if __name__ == '__main__':
    # txt文件所在路径
    txt_path = '/root/autodl-tmp/YOLO11/dataset/road_mark/test/labels'
    get_every_class_num(txt_path)