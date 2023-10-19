import os 
import pyautogui
import numpy as np
import cv2
from multiprocessing import Process
import concurrent.futures
import time 
from time import sleep
from convolutional_layer import ConvolutionalLayer
#This is going to either inherit from the Inference (Translator class)
#or be based on it

''' 
    Agent class handles taking an array of screenshots as a time-series input, 
    and feeding in the input to the Transformer to prompt the start of the transformer action.
    Using Pyautogui, the AI Agent should be in control of the game after having been prompted.
'''
class Agent:

    def __init__(self, context_window, img_width, img_height, d_model):
        self.img_width = img_width
        self.img_height = img_height
        self.context_window = context_window
        self.conv_layer = ConvolutionalLayer(img_width, img_height)
        self.d_model = d_model #Necessary for embeddings
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        #Imagine that our input screenshots were 28 * 28 pixel images (they're not).
        #If our queue is designed to hold the last N (context_window) images, 
        # then it must be initialized to be of dimension (context_window, 28, 28).
        # Until screenshots are recorded, they will all be zeroes.

        #We can use a numpy array to initialize the data structure dimensions but MUST typecase it as a list.
        #Possibly replace with the Queue data structure also built into Multiprocessing module.
        # self.queue = np.zeros(context_window * img_width * img_height).reshape(context_window, img_width, img_height).tolist()
        '''Replacing images with direct embeddings in the queue. Enables rapid computation of embeddings 
        as screenshots are taken, optimizing our exectu
        '''

        self.queue = np.zeros(context_window * d_model).reshape(context_window, d_model)#.tolist()
        # self.queue = []
        self.num_images = 0
    
    
    
    #Handle both the gameplay and the concurrent updates to our time series image data.
    def multiprocess(self, num_seconds):
        proc1 = Process(target=self.update_queue)  # instantiating without any argument
        proc2 = Process(target=self.play, args=(num_seconds,))

        proc1.start()
        proc2.start()

        proc1.join()
        proc2.join()


    def action_process_lifecycle(self, num_seconds):
        '''Let the AI Agent have control of the game for a maximum number of seconds.
        My friend was nice to lend me his rig for this mad science experiment, so I'll do 
        what I can to prevent a killer AI from taking control of his machine. 
        (I'M NOT JOKING. ALWAYS ENSURE THAT nu qm_seconds > 0 if trying this out for yourself, 
        or you will be fighting a bot for control of your keyboard).
        '''
        sleep(3) #This is T-5 seconds to AI Agent activation
        ##You want to update the image context even as the AI agent is playing, 
        ##so that any future decisions will be based on up-to-date information.
        ##Use multiprocessing module.
        # self.update_queue()
        # self.play(num_seconds)
        # self.multiprocess(num_seconds)
        count = 0 
        while count < num_seconds:
            self.pool.submit(self.update_queue)
            self.pool.submit(self.play)
            count+=1 
        self.pool.shutdown(wait=True)


    
    

    def play(self):
        '''Let the agent operate freely for a fixed period of num_seconds'''
        # start_time = time.time()
        # time.sleep(1)
        # current_time = time.time()
        # while current_time - start_time < num_seconds:
        '''Used the Oof test to ensure that the process exits the loop successfully.'''
        print('Oof')
        time.sleep(0.1)#test
        ######Put some actual code in here to automate the game.
        
        '''
        Take image queue data and feed it into transformer like so 
        #Reshape queue into image batch of size 1
            image_batch = np.array(self.queue).reshape(1, 
            self.context_window, 
            self.img_width, 
            self.img_height
        )
        #Retrieve embeddings 
        encoder_inputs = preprocess_img_embeddings(image_batch)
        #Where decoder inputs = padded array of null tokens (no action?)
        preds = transformer((encoder_inputs, decoder_inputs))
        
        predicted_token = np.argmax(preds) #Index corresponding to value of token.
        key_to_be_pressed = tokenizer.word_index(predicted_token)
        pyautogui.keydown(key_to_be_pressed) #This is where is gets fun!
        '''


            # current_time = time.time()
            # print(current_time - start_time)
            # pyautogui.keyDown('d')

    def convert_img_to_embedding(self, img):
        reshaped_img = img.reshape(1, 28, 28, 3) #Reshape to be input of batch_size=1
        #IMPORTANT: Figuring out that values had to be typecast as floats was a headache...
        reshaped_img = np.array(reshaped_img,  dtype='float32')
        embedding = self.conv_layer(reshaped_img)
        #For unit testing purposes, embeddings will be stored as numpy values 
        #although in production we can comment this line out and leave them as tensors.
        return embedding.numpy() 

    #Algorithm for managing time series screenshot data.
    #CHANGE: We're going to improve this method to upload not the images themselves, 
    #but rather they outputs of a convolutional layer.
    def enqueue(self, image):
        #Index of next image = num_images [1, 2, 0]
        if self.num_images < self.context_window:
            embedding = self.convert_img_to_embedding(image)
            #Enqueue the image embeddings for time series inference.
            self.queue[self.num_images] = embedding#image#embedding#image
            # self.queue.append(embedding)
            self.num_images+=1 #Update number of images saved to queue
        else:#If context window is full:
            self.queue.pop(0) #Remove oldest memeber of queue 
            embedding = self.convert_img_to_embedding(image)
            self.queue.append(embedding)
            #We don't update num_images because now it's guaranteed to constantly have 
            #num_images == context_window
    #Algorithm: 

    def update_queue(self):
        # Take a screenshot and feed it into the Convolutional Layer
        screencap = pyautogui.screenshot()
        screencap = np.array(screencap)#Load to queue ~
        screencap = cv2.resize(screencap, (self.img_width,  self.img_height))

        self.enqueue(screencap)#Add screencap to the queue 
        # sleep(1)
        # pyautogui.keyDown('d')
        # sleep(1)
        # pyautogui.keyUp('d')
        # print(len(self.queue),self.queue)
        '''Feed queue data structure into neural network every few seconds'''
    
agent = Agent(10, 28, 28, 32) #Queue will be a list of dimensions (10, 28, 28)
                          #The most recent 10 images of size 28 * 28; embeddings are of shape 32

#Simulate queue updates
# for i in range(3):
    # agent.update_queue()
# agent.play(5)
#OK so this works!
agent.action_process_lifecycle(2)
print(agent.queue)
# def main():
#     initializePyAutoGUI()
#     print("Done")


# def initializePyAutoGUI():
#     # Initialized PyAutoGUI
#     # When fail-safe mode is True, moving the mouse to the upper-left
#     # corner will abort your program.
#     pyautogui.FAILSAFE = True


# if __name__ == "__main__":
#     main()