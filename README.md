# Music-Art-and-LSTM-
This is typically a model used to generate a piece of music based on the given datasets.

To generate a good piece of music, we need model to understand what is good inside a music. To learn such sequence data, RNN is the most powerful architecture. With the help of Pytorch RNN module, we implemented RNN models with different sizes, types, and dropout values, as well as other factors like batch-size, optimizers, etc.. And finally, after training by Adagrad optimizer with batch-size being 50, a RNN containing one LSTM layer with 150 units, gave us the best validation loss, 1.617 and many beautiful musics. 

In order to get the best music according to our model, our team decide to do this part in the end, so we can get the best value for dropout, hidden unit and optimizer, we are going to discuss how we get these values in the rest part. The result we get is dropout=0.1, hidden units=150, numbers of layer=1, type of layer=LSTM, learning rate=0.001, batch-size=100,optimizer is Adagrad. 
