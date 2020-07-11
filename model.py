import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, p_dropout = 0.3):
        super(EncoderCNN, self).__init__()
        """
        EncoderCNN uses a pre-trained object detection CNN model to detect object features in color images. The classifier
        head of the original object detection CNN is stripped off and replaced by a new linear encoder layer, which encodes
        the detected image features as feature embeddings. The feature embeddings are then fed into another DecoderRNN to 
        generate image captions for the features detected by the Encoder CNN.
        """
        
        # Use a pre-trained ResNet50 model as feature detector. Strip-off the classifier head and replace it by a 
        # new linear encoder layer with embed_size outputs to produce embeddings of the detected image features
        
        # Obtain pre-trained ResNet50 feature detector model
        resnet = models.resnet50(pretrained=True)
        
        # Freeze model weights by switching off gradients for backpropagation
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Obtain the list of all layer modules and strip off the original classifier head (here: the last layer module)
        modules = list(resnet.children())[:-1]
        
        # Unpack the list of modules using '*'-operator and rebuild the model using nn.Sequential
        self.resnet = nn.Sequential(*modules)
        
        # Last module of the remaining ResNet50 feature detector: nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 
        # The last module of the remaining ResNet50 feature detector returns a stack of 1x1 feature maps. The number of 
        # feature maps in the adaptive average pooling layer corresponds to the number of feature maps of the previous 
        # convolutional layer. These 1x1 feature maps are then passed to a new feature embedding layer to obtain the 
        # the feature embeddings as input for the DecoderRNN.
        
        # Dropout layer
        self.dropout = nn.Dropout(p=p_dropout)
        
        # Encoding or feature embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
    
    
    def forward(self, images):
        """
        Forward propagation of the encoder CNN:
        Input arguments:
        :param images: Tensor of normalized color images of shape [batch_size, color_channels=3, width=224, height=224]
        Output arguments:
        :return features: Tensor of embedded image features of shape [batch_size, embed_size] as input to the DecoderRNN
        """
        # Obtain a batch of image features from the feature maps of the modified resnet feature detector model
        features = self.resnet(images)
        
        # Reshape the tensor of image features to shape [batch_size, embed_size]
        features = features.view(features.size(0), -1)
        
        # Generate feature embeddings as input for a subsequent DecoderRNN
        features = self.embed(self.dropout(features))
        
        # Return feature embeddings
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, p_dropout=0.3):        
        """
        Initialize the PyTorch DecoderRNN Module
        Input arguments:
        :param embed_size: Dimensionality of image and word embeddings as input to this RNN: The image embeddings are
        are obatained as encoded object detections from the Image EncoderCNN, and the word embeddings are given image 
        captions of previous steps (or are provided from ground truth annotations during training).
        :param hidden_size: The size of the hidden LSTM layer, which correpsonds to the number of LSTM outputs per time step
        :param vocab_size: The size of vocabulary, which corresponds to the number outputs of the linear word classifier 
        layer, which is connected to the output of the LSTM layer(s) and represents the final output of the overall RNN
        :optional param num_layers: Number of LSTM layers in the RNN (default = 1)
        :optional param p_dropout: Dropout probability used within the LSTM layer(s) if num_layers > 1 and in an
        additional dropout layer added between the LSTM layer(s) and the final linear decoder layer.
        """
        # Call constructor of base class to inherit its properties
        super(DecoderRNN, self).__init__()
        #super().__init__()
        
        # Set internal model properties
        self.embed_size = embed_size # output size of the embedding layer (= size of the word embedding vector)
        self.hidden_size = hidden_size # size of the hidden LSTM layer (= size of the LSTM output)
        self.vocab_size = vocab_size # input size of the embedding layer and output size of the fully-connected decoder layer
        self.num_layers = num_layers # number of hidden LSTM layers
        self.p_dropout = p_dropout # dropout probability (regularization)
        
        # Set device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        ## Define RNN model layers
        
        # Embedding layer (input to the LSTM containing the encoded object detections obtained from the Image EncoderCNN)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, 
                                       embedding_dim=embed_size)
        
        # Long Short-Term Memory (LSTM) layer
        if num_layers > 1:
            self.lstm = nn.LSTM(input_size=embed_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                dropout=p_dropout, 
                                batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=embed_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                dropout=0.0, 
                                batch_first=True)
        
        # Dropout layer between last LSTM layer and the subsequent fully connected linear word classifier layer
        self.dropout = nn.Dropout(p=p_dropout)
        
        # Final fully-connected linear word decoding layer to determine which word of the vocabulary is the most likely
        self.fc = nn.Linear(in_features=hidden_size, 
                            out_features=vocab_size, 
                            bias=True)
        
        # Since CrossEntropyLoss is used as optimization criterion there is no further softmax layer and
        # the scorses of the final fully-connected linear layer are output directly
        
        # Activation of the final fully-connected linear word decoding layer (only applied when predicting / sampling)
        # self.softmax = nn.Softmax(dim=1)
        
        # Remarks:
        # - If GPU is used the model is moved to the device together with its parameters in one step that needs
        #   to be initiated in the code before training or inference
        # - The hidden state and the cell state are initialized during training when the batch size is known!
        #   Here we only make a pre-initialization for batch_size = 1.
        # - Weights of embedding layer and fully-connected layer are initialized here.
        
        # Initialize the weights of the RNN decoder model's embeddding layer
        self.init_embeddings()
        
        # Initialize hidden state and cell state of the RNN decoder model's LSTM layer for default batch_size = 1
        _,_ = self.init_lstm_states()
        
        # Initialize weights and biases of the RNN's embeddding layer and fully-connected linear layer
        self.init_fc_weights()
        
        
    def init_embeddings(self):
        """
        Initialize the weights of the RNN decoder model's embedding layer:
        """
        # Set weights tensor of the RNN's embedding layer as random uniform
        self.embeddings.weight.data.uniform_(-0.1, 0.1).to(self.device)
        
    
    def init_fc_weights(self):
        """
        Initialize the weights and biases of the RNN decoder model's final fully-connected linear decoder layer:
        """        
        # Set bias tensor of the RNN's fully-connected linear layer to all zeros
        self.fc.bias.data.fill_(0.0).to(self.device)
        
        # Set weights tensor of the RNN's fully-connected linear layer as random uniform
        self.fc.weight.data.uniform_(-0.1, 0.1).to(self.device)
        
    
    def init_lstm_states(self, batch_size=1):
        """
        Initialize the hidden states and the cell states of the RNN decoder model's LSTM layer(s):
        
        At the start of training, we need to initialize both the hidden state and the cell state of the LSTM; 
        there will be none because the hidden state and the cell state are formed based on perviously seen data,
        which does not exist upon start of training as the LSTM has seen no data yet. So, this function defines 
        a hidden state and a cell state with all zeroes and an appropriate size (s. return argument).
        
        Input argument:
        :param batch_size: Batch size to be processed by the LSTM
        
        Output argument:
        :return (hidden_state, cell_state): Initial hidden state and cell state of the LSTM layer(s)
        
        The dimensions of hidden state and cell state are [num_layers, batch_size, hidden_size]
        """
        # The dimensions of hidden state and cell state are [num_layers, batch_size, hidden_size]
        #hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        #cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Get next set of weights by iterating to the next data set in self.parameters
        weight = next(self.parameters()).data
        
        # Create a tuple of two new tensors of shape [num_layers, batch_size, hidden_size] for the initial hidden 
        # state and the initial cell state of the LSTM layer, set them to zero, and move them to the device
        hidden_state = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device)
        cell_state = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device)
        
        # Return h0 and c0 as tuple of two tensors each of shape [num_layers, seq_len, hidden_size]
        return (hidden_state, cell_state)
    
    
    def forward(self, features, captions):
        """
        Forward propagation of the recurrent neural network (RNN):
        
        Input arguments:
        :param features: Tensor of embedded image features provided by the EncoderCNN of size [batch_size, embed_size]
        and type long() resp. int64
        :param captions: Tensor containing a batch of training sequences of tokenized image captions of size [batch_size, 
        seq_len] cast to type long() resp. int64, where seq_len may vary from batch to batch. The data loader (please
        refer to data_loader_train_val_test.py) takes care that each batch only contains captions of the same sequence 
        length and their corresponding images
        
        Output arguments:
        :return predicted_word_scores: Predicted word scores tensor of size [batch_size, captions.shape[1], vocab_size] 
        where captions.shape[1] corresponds to the word sequence length and predicted_word_scores[i,j,k] contains the 
        model's predicted word scores, indicating how likely the j-th token in the i-th caption in the batch is the 
        k-th token in the vocabular.
        """
        
        # Get batch size from EncoderCNN input
        batch_size = features.size(0)
        
        # Reshape the embedded image features tensor obtained from the EncoderCNN by adding another dimension for the
        # training sequence length.
        # Original features.shape = [batch_size, embed_size] => New feature.shape = [batch_size, seq_len, embed_size]
        embedded_features = features.unsqueeze(1)
        
        # Pass the training sequence of tokenized image captions (exepct for the last word) through the embedding layer
        # in order to obtain the training sequence of embedded captions of shape [batch_size, seq_len - 1, embed_size].
        # The first word in this input sequence is replaced by the embedded image features tensor (s. next step below)
        embedded_captions = self.embeddings(captions[:,:-1])
        
        # Concatenate the reshaped embedded image features tensor from the Image EncoderCNN as first "embedded word" 
        # and the training sequence of embedded image captions from the previous time steps (save the last word) along
        # the word sequence length axis (dim=1) to obtain the LSTM input vector.
        # The decoded output of the LSTM layer is then compared against the full target sequence of the image captions.
        lstm_input = torch.cat((embedded_features, embedded_captions), dim=1)
        
        # Initialize the hidden state and the cell state of the RNN decoder model's LSTM layer for each new batch on 
        # the device considering the actual batch_size. This way the history is detached from the LSTM model for each 
        # new batch.
        # hidden_state.shape = [num_layers, batch_size, hidden_size]
        # cell_state.shape = [num_layers, batch_size, hidden_size]
        (hidden_state, cell_state) = self.init_lstm_states(batch_size)
                
        # Pass the lstm_input and the initial hidden state and cell state through the LSTM layer(s)
        # lstm_output.shape = [batch_size, seq_len, vocab_size]
        lstm_output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
        
        # Stack up the lstm_output as contiguous tensor before passing the new tensor to the fully-connected linear 
        # decoder layer.
        # lstm_output.contiguous().shape = [batch_size * seq_len, vocab_size]
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_size)
        
        # Regularize the stacked lstm_output tensor using dropout and pass it on through the fully-connected decoder
        # layer to obtain a batch of raw unnormalized predicted word scorses
        # predicted_word_scores.shape = [batch_size * seq_len, vocab_size]
        predicted_word_scores = self.fc(self.dropout(lstm_output))
        
        # Reshape the raw unnormalized predicted word scores of shape [batch_size * seq_len, vocab_size] back into a 
        # tensor of shape [batch_size, seq_len, vocab_size], where seq_len may vary from batch to batch
        predicted_word_scores = predicted_word_scores.view(batch_size, -1, self.vocab_size)
        
        # As nn.CrossEntropyLoss() is used as loss function resp. criterion no LogSoftmax() is used here                
        return predicted_word_scores
    
    
    def argmax_sample(self, features, lstm_states=None, max_len=20, stop_word=1):
        """
        Generate a single sample caption (e. g. for testing purpose) on given image feature embeddings provided by 
        the EncoderCNN using argmax() to obtain the most likely image captions.
        
        Input arguments:
        :param features: LongTensor with embedded image features of a single test image provided by the EncoderCNN,
        whicht must be of shape [batch_size = 1, embed_size], where batch_size must be equal to 1!
        :optional param lstm_states: LongTensor tuple initial lstm states where lstm_states == (hidden_states, cell_states)
        with both hidden_states and cell_states must be of shape [num_layers, batch_size, hidden_size]
        :optional param max_len: maximum length of the predicted sample caption (default: max_len=20)
        :optional param stop_word: tokenized index of the stop word "<stop>" (default: stop_word=1)
        
        Output arguments:
        :return sample_caption: Generated sample caption for the test image of type list() starting with <start> 
        (start_word=0 => automatically generated by the Decoder RNN) and ending with <stop> (stop_word=1) 
        """
        
        # Get batch size from EncoderCNN input and check if it is 1
        batch_size = features.size(0)
        assert (batch_size == 1), "Batch size must be set to batch_size = 1 in validation mode!"
        
        # Reshape the embedded image features tensor obtained from the EncoderCNN by adding another dimension for the
        # training sequence length.
        # Old features.shape = [batch_size=1, embed_size]
        # New feature.shape = [batch_size=1, seq_len=1, embed_size]
        embedded_features = features.unsqueeze(1)
        
        # Initialize the lstm_states = (hidden_state, cell_state) of the RNN decoder model's LSTM layer(s) with zeros
        # if not provided from externally where hidden_states and cell_states must be of shape ...
        # hidden_states.shape = [num_layers, batch_size=1, hidden_size]
        # cell_states.shape = [num_layers, batch_size=1, hidden_size]
        if lstm_states == None:
            # Automatically initialize lstm_states = (hidden_states, cell_states) with zeros
            (hidden_states, cell_states) = self.init_lstm_states(batch_size)
        else:
            # Initialize lstm_states = (hidden_states, cell_states) with externally provided values
            print('lstm_states provided from externally ...')
            (hidden_states, cell_states) = lstm_states
        # Check sizes of lstm_states
        assert hidden_states.shape == torch.Size([self.num_layers, batch_size, self.hidden_size]), \
            "hidden_states must be of shape [num_layers, batch_size, hidden_size]"
        assert cell_states.shape == torch.Size([self.num_layers, batch_size, self.hidden_size]), \
            "cell_states must be of shape [num_layers, batch_size, hidden_size]"
        
        # Initialize sample caption as empty list
        sample_caption = list()
        
        # Loop over max_len time steps to predict an image caption sequence. When the sequence is completed by adding
        # a <stop> word or when maximum sequence length max_len is reached, the loop is interrupted.
        for t in range(max_len):
            
            # For the first time step t == 0 the input to the lstm layer is the feature vector
            if t == 0:
                # At the first time step pass the embedded features tensor and the lstm_states initialized with zeros 
                # through the lstm layer(s). This way, the first predicted word will automatically be the <start> word.
                lstm_output, (hidden_states, cell_states) = self.lstm(embedded_features, (hidden_states, cell_states))
                
            # For any further time step t > 0 the input to the lstm layer is the embedded next predicted word
            else:
                # At all subsequent time steps pass the predicted word from the previous time step and the updated 
                # lstm_states through the lstm layer(s) to obtain the lstm_output for the next word prediction
                lstm_output, (hidden_states, cell_states) = self.lstm(embedded_prev_word, (hidden_states, cell_states))
            
            # Decode the lstm_output of shape [batch_size=1, seq_len=1, embed_size] to obtain the word scores of the 
            # predicted word (last word in lstm_output) of shape [batch_size=1, vocab_size]
            predicted_word_scores = self.fc(lstm_output[:, -1, :])
            
            # Get the predicted word using argmax()
            predicted_word = torch.argmax(predicted_word_scores, dim=1)
                        
            # Training needs to include the <stop> word index (default: 1), otherwise it won't be emitted.
            if predicted_word.item() == stop_word:
                # Add the last predicted word to sample caption and break the loop
                sample_caption.append(predicted_word.item())
                break
            elif t == max_len-1:
                # If maximum sequence length is reached add the <stop> word index as last word and break the loop
                sample_caption.append(stop_word)
                break
            else:
                # Add the predicted word to the sample caption
                sample_caption.append(predicted_word.item())
                
                # Get the embeddings of the last predicted word to prepare for the next cycle and add again another 
                # dimension for the sequence length by unsqueezing along axis dim=1.
                # Old predicted_word.shape = [batch_size=1, embed_size]
                # New predicted_word.shape = [batch_size=1, seq_len=1, embed_size]
                embedded_prev_word = self.embeddings(predicted_word.unsqueeze(1))
        
        # Return the predicted sample image caption                
        return sample_caption


    def beamsearch_sample(self, features, lstm_states=None, beam_size=3, max_len=20, stop_word=1, show_alternatives=False):
        """
        Modified Beam Search approach inspired by
        
            https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py
        
        Generate a single sample caption (e. g. for testing purpose) on given image feature embeddings provided by
        the EncoderCNN using Beam Search to obtain the most likely image captions.
        
        Batched Beam Search does exactly this:
        - At the first decode step, consider the top k predicted word candidates.
        - Generate k different second word candidates for each of these k first words (by looking at the top k * k ranked
          words, chosing the k different ones and omitting all duplicates).
        - Choose the top k [first word, second word] combinations considering additive scores normalized by the sequence 
          length to penalize for too long sequences e. g. with repetitions of words.
        - For each of these k second words, choose k third words, choose the top k [first word, second word, 
          third word] combinations (s. above).
        - Repeat at each decode step.
        - After k sequences terminate, choose the sequence with the best overall score.
        - Stop each sequence resp. caption sentence if <stop> is reached or max_len is reached.
        - Terminate each sequence with the <stop> word as last word.
        
        Input arguments:
        :param features: LongTensor with embedded image features of a single test image provided by the EncoderCNN,
        whicht must be of shape [batch_size = 1, embed_size], where batch_size must be equal to 1!
        :optional param lstm_states: LongTensor tuple initial lstm states where lstm_states == (hidden_states, cell_states)
        with both hidden_states and cell_states must be of shape [num_layers, batch_size, hidden_size]
        :optional param beam_size: beam size parameter for the beam search (default: beam_size=3), which determines
        the number of top k predictions to be considered for the combinations with the top k sentences
        :optional param max_len: maximum length of the predicted sample caption (default: max_len=20)
        :optional param stop_word: tokenized index of the stop word "<stop>" (default: stop_word=1)
                
        Output arguments:
        :return sample_caption: Generated sample caption for the test image of type list() starting with <start> 
        (start_word=0 => automatically generated by the Decoder RNN) and ending with <stop> (stop_word=1)
        """
        
        ### Batched Beam Search ###
        # Do not use a batch_size > 1 for batch_size will be set equal to beam_size!
        
        # Get batch size from EncoderCNN input and check if it is equal to 1
        batch_size = features.size(0)
        assert (batch_size == 1), "Batch size must be set to batch_size = 1 in validation mode!"
        
        # Lists to store completed sequences and probability scores
        complete_seqs = list()
        complete_seqs_scores = list()
        
        # Reshape the embedded image features tensor obtained from the EncoderCNN by adding another dimension for the
        # training sequence length.
        # Old features.shape = [batch_size=1, embed_size]
        # New feature.shape = [batch_size=1, seq_len=1, embed_size]
        embedded_features = features.unsqueeze(1)
        
        # Treat this problem as having a batch size equal to beam size k (batch_size == beam_size)
        k = beam_size
        
        # Stack up embedded features tensor k-times (batch_size == beam_size)
        # embedded_features.shape = [batch_size=k, seq_len=1, embed_size]
        embedded_features = embedded_features.repeat(k, 1, 1)
        
        # Initialize the lstm_states = (hidden_states, cell_states) of the RNN decoder model's LSTM layer(s) with zeros
        # if not provided from externally where hidden_states and cell_states must be of shape ...
        # hidden_states.shape = [num_layers, batch_size=k, hidden_size]
        # cell_states.shape = [num_layers, batch_size=k, hidden_size]
        if lstm_states == None:
            # Automatically initialize lstm_states = (hidden_states, cell_states) with zeros
            (hidden_states, cell_states) = self.init_lstm_states(batch_size=beam_size)
        else:
            # Initialize lstm_states = (hidden_states, cell_states) with externally provided values
            print('lstm_states provided from externally ...')
            (hidden_states, cell_states) = lstm_states
        # Check sizes of lstm_states
        assert hidden_states.shape == torch.Size([self.num_layers, beam_size, self.hidden_size]), \
            "hidden_states must be of shape [num_layers, batch_size, hidden_size]"
        assert cell_states.shape == torch.Size([self.num_layers, beam_size, self.hidden_size]), \
            "cell_states must be of shape [num_layers, batch_size, hidden_size]"
        
        # Loop over max_len time steps to predict k sequences of image captions. When sequences are completed by adding
        # a <stop> word or when maximum sequence length max_len is reached, they are taken out and the batch size resp.
        # beam_size is reduced to batch_size == beam_size == s with s <= k
        for t in range(max_len):
            
            # For the first time step t == 0 the input to the lstm layer is the feature vector
            if t == 0:
                # At the first time step pass the embedded features tensor and the lstm_states initialized with zeros 
                # through the lstm layer(s). This way, the first predicted word will automatically be the <start> word.
                lstm_output, (hidden_states, cell_states) = self.lstm(embedded_features, (hidden_states, cell_states))
                
                # Decode the lstm_output of shape [batch_size=s, seq_len=1, embed_size] to obtain the word scores of 
                # the first predicted word (last word in lstm_output) of shape [batch_size=k, vocab_size].
                scores = nn.functional.softmax(self.fc(lstm_output[:, -1, :]), dim=1) # (k, vocab_size)
                
                # Initialize tensors to store the top k cumulated probability scores of all k previously predicted words 
                # and the k predicted words themselves (= indices of the maximum values): now both tensors contains just
                # (k) scores for the initially predected <start>_words, which should be nearly 1.0.
                top_k_seqs_scores, k_prev_words = torch.max(scores, dim=1) # (k), (k)
                
                # Unsqueeze top_k_seqs_scores and k_prev_words along axis dim=1 to add another dimension for sequence length
                top_k_seqs_scores = top_k_seqs_scores.unsqueeze(1) # (k, 1)
                k_prev_words = k_prev_words.unsqueeze(1) # (k, 1)
                
                # Initialize tensor to store top k sequences; now it contains just the <start> words predicted in the 
                # first time step.
                top_k_seqs = k_prev_words  # (k)
                
                # Pass the k previously predicted words of shape [batch_size=k] through the embedding layer and add 
                # another dimension for the sequence length by unsqueezing along axis dim=1 in order to obtain the 
                # word embeddings of shape [batch_size=k, seq_len=1, embed_size] for the next step.
                # k_prev_words.shape = [batch_size=k]
                # k_prev_words.unsqueeze(1).shape = [batch_size=k, seq_len=1]
                # embedded_k_prev_words.shape = [batch_size=1, seq_len=1, embed_size]
                embedded_k_prev_words = self.embeddings(k_prev_words) # (k, 1, embed_size)
                
            # For any further time step t > 0 and t < max_len the input to the lstm layer are the embedded k (resp. s)
            # previous words
            else:
                # Proceed only if maximum sequence length has not yet been reached
                if t == max_len:
                    # For the last step t == max_len do not make any further predictions and complete all remaining
                    # incomplete sequences by adding a <stop> word while keeping the top k sequence scores of the 
                    # previous steps as they are (no further updates)
                    complete_seqs.extend(torch.cat((top_k_seqs[incomplete_inds], 
                                                    stop_word*torch.ones(k, 1)), dim=1).cpu().tolist()) # (k, seq_len)
                    complete_seqs_scores.extend(top_k_seqs_scores[incomplete_inds].cpu().tolist()[:]) # (k, seq_len)
                    
                    # Break the loop after completing the rest of the sequences
                    break
                                
                # At all subsequent time steps pass the k predicted words from the previous time step and the updated
                # lstm_states through the lstm layer(s) to obtain the lstm_output for the next word prediction.
                lstm_output, (hidden_states, cell_states) = self.lstm(embedded_k_prev_words, (hidden_states, cell_states))
                
                # Decode the lstm_output of shape [batch_size=s, seq_len=1, embed_size] to obtain the word scores of the 
                # next predicted word (last word in lstm_output) of shape [batch_size=s, vocab_size].
                scores = nn.functional.softmax(self.fc(lstm_output[:, -1, :]), dim=1) # (s, 1)
                
                # Add probability scores of the top k words predicted in this step to the cumulated probability scores
                # of the previously predicted words stored in the top k sequences                
                scores = top_k_seqs_scores.expand_as(scores) + scores # (s, vocab_size)
                
                # Unroll the tensor of next word probabilities and find the top k candidates and their unrolled indices
                # If you fed back the only the top k candidates from top k predictions you would end up with k time the
                # same result. In this case beam search would produce the same results as argmax search. Therefore, first
                # find the top k * k candidates and enforce to filter for the top k different ones, which are not yet in
                # the list of the top k candidates.
                top_k_scores, top_k_words = scores.view(-1).topk(k=k*k, dim=0, largest=True, sorted=True) # (s), (s)
                
                # Go through the list of k * k top candidates and chose only those top k candidates, which are different
                # from one another with respect to the vocabulary
                selected_words = []
                selected_inds = []
                len_selected_inds = 0
                for j in range(k*k):
                    if (top_k_words[j] % self.vocab_size) not in selected_words:
                        selected_words.append(top_k_words[j] % self.vocab_size)
                        selected_inds.append(j)
                        len_selected_inds += 1
                    if len_selected_inds == k:
                        top_k_scores = top_k_scores[selected_inds]
                        top_k_words = top_k_words[selected_inds]
                        break
                
                # Convert the top k unrolled word indices to actual sequence indices and next word indices, which indicate the
                # the top k combination of sequences and next words that will be kept for further processing.
                # Range of sequence indices: seq_inds[i] = 0...s-1 for i = 0...s-1
                # Range of next word indices: next_word_inds[i] = 0...vocab_size-1 for i = 0...s-1
                seq_inds = top_k_words // self.vocab_size  # (s)
                next_word_inds = top_k_words % self.vocab_size  # (s) # (s)
                
                # Unsqueeze top_k_scores tensor with shape [batch_size=s] along axis dim=1 and update the tensor of 
                # cumulative sequences scores with shape [batch_size=s, 1]
                top_k_seqs_scores = top_k_scores.unsqueeze(1) # (s, 1)
                
                # Unsqueeze next_word_inds with shape [batch_size=s] along axis dim=1 and update k_prev_words tensor
                # with shape [batch_size=s, seq_len=1]
                k_prev_words = next_word_inds.unsqueeze(1) # (s, 1)
                
                # Add the predicted next words to the sequences - keeping only the top k best overall combinations
                # top_k_seqs.shape = (s, seq_len=t+1)
                top_k_seqs = torch.cat([top_k_seqs[seq_inds], k_prev_words], dim=1)  # (s, seq_len+1)
                
                # Check which sequences are incomplete resp. didn't reach the <stop> word? They will be further processed
                incomplete_inds = [idx for idx, next_word in enumerate(next_word_inds) if next_word != stop_word] # (s)
                # Remove the set of indices for the incomplete sequences from the range of possible indices to obtain the
                # remaining set of incides for the complete sequences
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds)) # (k - s)
                
                # Set aside complete sequences (sequence lengths may be different for all completed sequences)
                if len(complete_inds) > 0:
                    #complete_seqs.extend(top_k_seqs[complete_inds].tolist()) # (k - s) with individual sequence legnths
                    #complete_seqs_scores.extend(top_k_scores[complete_inds]) # (k - s)
                    complete_seqs.extend(top_k_seqs[complete_inds].cpu().tolist()) # (k, k - s)
                    complete_seqs_scores.extend(top_k_seqs_scores[complete_inds].cpu().tolist()[:]) # (k, k - s)
                k -= len(complete_inds)  # reduce beam length k resp. s accordingly
                
                # Proceed only with incomplete sequences
                if k == 0:
                    # Break if all sequences have been completed
                    break
                    
                # Keep only incomplete sequences and prepare the batches for the next iteration
                top_k_seqs = top_k_seqs[incomplete_inds]
                
                # Keep only those elements that corresponds to incomplete sequences in hidden state and cell state
                hidden_states = hidden_states[:, seq_inds[incomplete_inds], :]
                cell_states = cell_states[:, seq_inds[incomplete_inds], :]
                
                # Keep only the scores of the incomplete sequences
                top_k_seqs_scores = top_k_seqs_scores[incomplete_inds]
                
                # Keep only the next words of the incomplete sequences, which are not a <stop> word
                k_prev_words = k_prev_words[incomplete_inds]
                
                # Pass the k previous tokenized words of shape [batch_size=s, seq_len=1] through the embedding layer 
                # to obtain the word embeddings of shape [batch_size=s, seq_len=1, embed_size] for the next step.
                embedded_k_prev_words = self.embeddings(k_prev_words)
        
        # Normalize completed sequences by sequence length to penalize longer captions
        normalized_complete_seqs_scores = list()
        for j in range(len(complete_seqs_scores)):
            normalized_complete_seqs_scores.extend([complete_seqs_scores[j][0]/len(complete_seqs[j])])
        
        # Get predicted sequence with the maximum total score normalized by the individual sequence length
        #i = complete_seqs_scores.index(max(complete_seqs_scores)) # use additive scores
        i = normalized_complete_seqs_scores.index(max(normalized_complete_seqs_scores)) # use normalized additive scores
        sample_caption = complete_seqs[i]
        
        # Show alternative sequences if show_alternatives == True
        if show_alternatives == True:
            print('Show alternatives:')
            print('Complete sequence scores      = ', complete_seqs_scores)
            print('Normalized compl. seq. scores = ', normalized_complete_seqs_scores)
            print('Complete sequences            = ', complete_seqs)
        
        # Return predicted sample caption with the highest cumulative probability score
        return sample_caption
    