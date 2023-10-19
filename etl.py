import os 
import cv2
import numpy as np
#Set environment variables
FOOTAGE = 'FOOTAGE'#Path to footage folder
KEYLOG = 'KEYLOG'#Path to keylog.txt

os.environ["FOOTAGE"] = "/home/dolan/Portfolio/AutopilotAI/automation/data_pipeline/footage/"
os.environ["KEYLOG"] = "/home/dolan/Portfolio/AutopilotAI/automation/data_pipeline/keylog.txt"
# os.environ["KERAS_BACKEND"] = "tensorflow"


print(os.environ.get('FOOTAGE'))


def read_keylog():
    # Using readlines()
    filename = os.environ.get('KEYLOG')
    file_ = open(filename, 'r')
    Lines = file_.readlines()
    
    keystrokes = []

    count = 0
    # Strips the newline character
    for i in range(len(Lines)):
        count += 1
        # current_line = Lines[i].strip().split('-')
        # print("Line{}: {}".format(count, current_line))
        #Every 5 lines corresponds to an image frame associated with that keystroke. 
        if count % 5 == 0:
            keystroke_line = Lines[i - 4].strip().split('-')
            keystrokes.append(keystroke_line[-1])

    return keystrokes 





def read_images():
    dir_ = os.environ.get('FOOTAGE')
    img_array = []
    img_dictionary = {}
    for i in range(len(os.listdir(dir_))):
        try:

            filename = os.listdir(dir_)[i]
            file_count = filename.split('_')[1].split('.')[0]
            # print(int(file_count))
            print(int(file_count))
            img_name = dir_ + os.listdir(dir_)[i]
        # for img in dir_:
            # img_name = dir_  +  img
            # print(img_name)
            img = cv2.imread(img_name)
            # print(img.shape)
            # img_array.append(img)
            img_dictionary[file_count] = img
        except:
            print(img)
    
    #Since the integer value of filecount indicates what order the screenshot was saved in, 
    #we can sort the values by key in order to retrieve the correct img_array in order of live recording. 
    img_dictionary = dict(sorted(img_dictionary.items()))
    img_array = list(img_dictionary.values())
    # return img_array, img_dictionary
    return img_array

img_array = read_images()
keys = read_keylog()
# print(keys)

img_array.pop() #There is an extra key from pressing control C to end the program- this 
img_array, keys = img_array[:len(img_array) - 2], keys[:len(keys)-2]
print(len(img_array), len(keys))


def chunk_sequence(stream, chunk_size):
    sequences = []
    i = 0 

    while i < len(stream):
        current_sequence = stream[i * chunk_size: (i + 1) * chunk_size]
        sequences.append(current_sequence)
        i+=1 
    return sequences
#Write a data pipeline to filter data from footage and place it into a set of batches 
#of the form [images, keystrokes]
#specify a chunk_size into which to organize each sample sequence
#organize groups of sequences into batches )
def pipeline(image_context_length, keypress_sequence_length):
    img_array = read_images()
    keys = read_keylog()
    # print(len(img_array), len(keys))
    #Remove first img, so that every image precedes the corresponding keystroke.
    img_array.pop(0) #There is an extra key from pressing control C to end the program- this 
    #causes there to be one extra img in img_array    
    img_array, keys = img_array[:len(img_array) - 2], keys[:len(keys)-2]
    # print(len(img_array), len(keys))
    #We want all keystrokes that come before their images.
    '''Of the form 
    [[img1, img2,...],[img1, img2,....],...]
    '''
    img_sequences = chunk_sequence(img_array, chunk_size)
    
    key_sequences = chunk_sequence(keys, chunk_size)
    

    return img_sequences, key_sequences 

#Define categorical labels from time series tokens.
def define_labels(key_tokens, end_token):#Takes in array of integer tokens, and generates decoder target values
    # end_token=0#Tokenize END
    labels = key_tokens[:]
    labels.pop(0)
    #The end token 
    labels.append(end_token)
    
    return labels 

#Pad key_tokens, labels
# vec = np.array([1,2,3])
# vec_padded = np.pad(vec,(0, 7), 'constant')
# print(vec_padded)

# img_sequences = chunk_sequence(img_array, 10)
# print(img_sequences[:20])