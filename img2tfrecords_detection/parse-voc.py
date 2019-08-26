import numpy as np
import glob
import xml.etree.ElementTree as ET
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
index_map = dict(zip(CLASSES, range(len(CLASSES))))
print(index_map)
xml_file_list=glob.glob("/mnt/westerndatadisk/datasets/pascal_voc/VOC2012test/Annotations/*.xml")
with open('voc2012_test_bbox.txt','w',encoding='utf-8') as f:
    for xml in xml_file_list:
        tree = ET.parse(xml)
        root = tree.getroot()
        '''
        输出所有子树的根节点名字（相当于keys）
        for child in root:
            print(child.tag)
        '''
        filename=(root.find('filename').text)

        '''
        root.findall('str'):返回str代表的所有子树列表
        root.iter('str'):迭代查找str代表的子树
        root.getchildren():返回所有子树列表
        '''
        objects=root.findall('object')
        if len(objects)==0 or objects is None:
            continue
        label=""
        label += filename
        label += " "
        obj_num=len(objects)
        for i in range(obj_num):
            obj=objects[i]
            classname=obj.find('name').text
            classid=index_map[classname]
            # print(classname,classid)
            label+=str(classid)
            label += " "
            bndbox_root=obj.find('bndbox')
            x_min=bndbox_root.find('xmin').text
            y_min = bndbox_root.find('ymin').text
            x_max = bndbox_root.find('xmax').text
            y_max = bndbox_root.find('ymax').text
            # print(x_min,y_min,x_max,y_max)
            label+=x_min
            label += " "
            label +=y_min
            label += " "
            label+=x_max
            label += " "
            label+=y_max
            if i!=(obj_num-1):
                label += " "
        f.write(label)
        f.write('\n')
        print(xml)
