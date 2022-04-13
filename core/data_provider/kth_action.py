import numpy as np
import os
import cv2
from PIL import Image
import logging
import random

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices,category, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.category=category
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 1)).astype(
            self.input_data_type)
        input_category=np.zeros(self.minibatch_size)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            input_category[i]=self.category[begin]
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch,input_category

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))

class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
        self.category_2 = ['jogging', 'running']
        self.category = self.category_1 + self.category_2
        self.image_width = input_param['image_width']

        self.train_person = ['01', '02', '03', '04', '05', '06', '07', '08',
                             '09', '10', '11', '12', '13', '14', '15', '16']
        self.test_person = ['17', '18', '19', '20', '21', '22', '23', '24', '25']

        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    # Specific_category is for Continual learning for different categories
    def load_data(self, paths, mode='train', specific_category = None):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        path = paths[0]
        person_id = []
        if mode == 'train':
            person_id = self.train_person
        elif mode == 'test':
            person_id = self.test_person
        else:
            print("ERROR!")
        if specific_category == None:
            specific_category = self.category

        print('begin load data' + str(path))

        frames_np = []
        frames_file_name = []
        frames_person_mark = []
        frames_category = []
        person_mark = 0
        frames_specific_category=[]
        # c_dir_list = self.category
        c_dir_list = specific_category

        frame_category_flag = -1
        for i,c_dir in enumerate(c_dir_list): # handwaving
            if c_dir in self.category_1:
                frame_category_flag = 1 # 20 step
            elif c_dir in self.category_2:
                frame_category_flag = 2 # 3 step
            else:
                print("category error!!!")

            c_dir_path = os.path.join(path, c_dir)
            p_c_dir_list = os.listdir(c_dir_path)

            for p_c_dir in p_c_dir_list: 
                if p_c_dir[6:8] not in person_id:
                    continue
                person_mark += 1
                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist.sort() 
                for file in filelist: 
                    if file.startswith('image') == False:
                        continue
                    # print(file)
                    # print(os.path.join(dir_path, file))
                    frame_im = Image.open(os.path.join(dir_path, file))
                    frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                    # print(frame_np.shape)
                    frame_np = frame_np[:, :, 0] #
                    frames_np.append(frame_np)
                    frames_file_name.append(file)
                    frames_person_mark.append(person_mark)
                    frames_category.append(frame_category_flag)
                    frames_specific_category.append(i)
        # is it a begin index of sequence
        indices = []
        index = len(frames_person_mark) - 1
        while index >= self.seq_len - 1:
            if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
                end = int(frames_file_name[index][6:10])
                start = int(frames_file_name[index - self.seq_len + 1][6:10])
                # TODO: mode == 'test'
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    if frames_category[index] == 1:
                        index -= self.seq_len - 1
                    elif frames_category[index] == 2:
                        index -= 2
                    else:
                        print("category error 2 !!!")
            index -= 1

        frames_np = np.asarray(frames_np)
        data = np.zeros((frames_np.shape[0], self.image_width, self.image_width , 1))
        for i in range(len(frames_np)):
            temp = np.float32(frames_np[i, :, :])
            data[i,:,:,0]=cv2.resize(temp,(self.image_width,self.image_width))/255
        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices,frames_specific_category

    def get_train_input_handle(self, specific_category = None):
        train_data, train_indices,train_category = self.load_data(self.paths, mode='train', specific_category = specific_category)
        return InputHandle(train_data, train_indices,train_category, self.input_param)

    def get_test_input_handle(self, specific_category = None):
        test_data, test_indices,test_category = self.load_data(self.paths, mode='test', specific_category = specific_category)
        return InputHandle(test_data, test_indices, test_category,self.input_param)

