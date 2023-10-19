# AutopilotAI-2
A new spin on AutopilotAI, replacing the recurrent LSTM with Transformers and adding Image Embeddings.

##  What Is This? 

As a former CTO and a software engineer, I am constantly surrounded by the hype of Generative AI for mundane applications like work, school, marketing, etc.
I dug out one of my old portfolio projects on an old self-driving project (AutopilotAI) and decided to put my expert-level skills to the test and rebuild the original project on top of a Causal Transformer architecture.

Unlike the previous AutopilotAI project (version 1), this version can now not only drive in GTA 5 and similar RPG's- it can walk, fight, shoot, and perform 
any physical action in-game that a live human player can.

My motivation for the project is twofold-
  ## 1. Demonstrate my advanced software/ML engineering skills and 
  ## 2. Show the world that Generative AI has both enormous potential for robotics applications, in addition to extreme physical risks should the former 
  ## point be truly realized. 
  ## This is a killer robot designed to work in videogames. If I could build it in under a month by myself, imagine what a rogue state could do.
  ## (Food for thought!) :)

  
There are 3 key branches:

main- contains README and a high-level overview of the project.
data_pipeline- Recording software for obtaining, sorting, and preprocessing training data 
training- contains folder to train and validate model for future use. 
inference- deployment of live neural network via an Agent class, using ThreadPooling to concurrently 
update the state of the image history context and generate predictions on new data. 

This is a capstone software and machine learning engineering project, which took exactly 2 weeks to the date of this README (Oct 05-19) 
to build. Various aspects of software design and engineering come into play in the building of this project:
  ## Concurrency- Use of executing multiple programs across different threads simultaneously in order to ensure a reliable, robust AI Agent 
  ## Expert Use of Design Patterns- Aspects of the model employ various design patterns efficiently, such as using the Factory Design Pattern to implement neural network architectures with slight changes in the inherited parent class. The Decorator Design Pattern was used with Tensorflow in order to build and modify a custom train_step() function, for the purposes of disabling eager (greedy) execution in the Tensorflow model. 
  ## Expert Use of Object-Oriented Programming concepts- Polymorphism, Encapsulation, Abstraction and Inheritance were used heavily throughout this project, in addition to the employment of SOLID Principles (Single-Responsibility, Open-Closed, Liskov Distribution, Interface Segregation and Dependency-Inversion)
  ## Advanced understanding of Deep Learning- In addition to building on top of Tensorflow's implementation of **Attention Is All You Need**, I wrote a custom training loop using masked-self attention in order to perform backpropagation from the output layer of the decoder back through the rest of the model. Additionally, I made changes to the original architecture of the Transformer from the original paper, replacing the Embedding layer of the Encoder with a pretrained Convolutional Neural Network, originally trained on the COCOA-80 image dataset.
  ##  Feature Engineering Skills- Used aforementioned pretrained Convolutional Layer as a feature engineering step, in order to capture image embeddings directly from screenshots of live gameplay footage.
  ## Containerization- Usage of Docker containers to ensure that software written on my Linux machine can port over efficiently to a Windows machine with an NVIDIA GPU in order to profit immensely from the performance boost during training and inference. 
  ## Data Engineering- Expert use of building live data pipelines for collecting and sorting time-series training data for a live application.

For those of you who take a look at this, I hope you're able to have fun with the application. Don't rely on it too much- it's probably not an insanely good player- yet!

  

  
