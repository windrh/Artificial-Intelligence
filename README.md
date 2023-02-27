# Artificial-Intelligence
This project was a personal challenge that has now taken over two hundred hours of learning every aspect of modern Artificial Intelligence from scratch, and then implementing this knowledge in code. The "challenge" part of this project is the fact I am only allowing myself to import math, random, and copy modules. This will ensure that I have pride in my product and can demonstrate my mastering of the concepts.  

This project's end goal is generally to have a final software that can learn. The furthest client side development will only accomadate for commands entered into a python console or command line (if compiled). I also intend to develop a documentary of code for the Functions.py. 

This is in the middle phase of development, with the backend work for backpropagation being the current phase I am working through. 

## Functions 
This folder is a module in itself for the client end program to use for actual A.I. development and manipulation inside of an application. 


# Update History 

Release Version - Date Released - Brief Description of the Release 

Release 0.1.0 - 2/14/2023 - First code is uploaded into Github, includes methods for my A.I. model variables, working forward propagation, and in-development          backpropagation. 

Release 0.1.1 - 2/16/2023 - Updates the comments of backpropagation in order to show what is happening on an abstract level with my "Node" class. 

Release 0.1.2 - 2/22/2023 - FUNCTION(backprop) updated: a single iteration of nodes can now be developed successfully based on the model. This is critical for finding derivatives dynamically. 

Release 0.2.0 - 2/26/2023 - FUNCTION(backprop) UPDATED: now working. For any range and training alpha value (how slow or fast you want the network to be trained), backprop will find the gradient and apply it to update weights. Effectively, the hardest part of the project is done and now the rest of it is contingent on implementing fixes for minute errors within the realm of A.I. Known Bug: iterations of the gradient do not save into universal model for some reason. This means that the calculations and correct model is being found but for some reason it is not being saved for future backpropogation iterations to use. bug will be fixed later releases.
