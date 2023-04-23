Download Link: https://assignmentchef.com/product/solved-assignment-2-recurrent-neural-networks-and-graph-neural-networks
<br>
In this assignment you will study and implement recurrent neural networks (RNNs) and have a theoretical introduction to graph neural networks (GNNs). Recurrent neural networks are best suited for sequential processing of data, such as a sequence of characters, words or video frames. Their applications are mostly in neural machine translation, speech analysis and video understanding. These networks are very powerful and have found their way into many production environments. For example <a href="https://research.google.com/pubs/pub45610.html">Google’s neural </a><a href="https://research.google.com/pubs/pub45610.html">machine translation system</a> relies on Long-Short Term Networks (LSTMs). Graph Neural Networks are specifically applied to graph-structured data, like knowledge graphs, molecules or citation networks.

The assignment consists of three parts. First, you will get familiar with vanilla RNNs and LSTMs on a simple toy problem. This will help you understand the fundamentals of recurrent networks. After that, you will use LSTMs for learning and generating text. In the final part, you will analyze the forward pass of a graph convolutional neural network, and then discuss tasks and applications which can be solved using GNNs. In addition to the coding assignments, the text contains multiple questions which you need to answer. We expect each student to hand in their code and individually write a report that explains the code and answers the questions.

Before continuing with the first assignment, we highly recommend each student to read this excellent blogpost by Chris Olah on recurrence neural networks: <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding </a><a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">LSTM Networks</a><a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">.</a> For the second part of the assignment, you also might want to have a look at the <a href="https://pytorch.org/docs/stable/nn.html#recurrent-layers">PyTorch recurrent network documentation</a><a href="https://pytorch.org/docs/stable/nn.html#recurrent-layers">.</a>

<h1>1           Vanilla RNN versus LSTM    (Total: 50 points)</h1>

For the first task, you will compare vanilla Recurrent Neural Networks (RNN) with LongShort Term Networks (LSTM). PyTorch has a large amount of building blocks for recurrent neural networks. However, to get you familiar with the concept of recurrent connections, in this first part of this assignment you will implement a vanilla RNN and LSTM from scratch. The use of high-level operations such as torch.nn.RNN, torch.nn.LSTM and torch.nn.Linear is not allowed until the second part of this assignment.

<h2>1.1           Toy Problem: Palindrome Numbers</h2>

In this first assignment, we will focus on very simple sequential training data for understanding the memorization capability of recurrent neural networks. More specifically, we will study <em>palindrome </em>numbers. Palindromes are numbers which read the same backward as forward, such as:

303

4224

175282571

682846747648286

We can use a recurrent neural network to predict the next digit of the palindrome at every timestep. While very simple for short palindromes, this task becomes increasingly difficult for longer palindromes. For example when the network is given the input 68284674764828_ and the task is to predict the digit on the _ position, the network has to remember information from 14 timesteps earlier. If the task is to predict the last digit only, the intermediate digits are irrelevant. However, they may affect the evolution of the dynamic system and possibly erase the internally stored information about the initial values of input. In short, this simple problem enables studying the memorization capability of recurrent networks.

For the coding assignment, in the file part1/dataset.py, we have prepared the class PalindromeDataset which inherits from torch.utils.data.Dataset and contains the function generate_palindrome to randomly generate palindrome numbers. You can use this dataset directly in PyTorch and you do not need to modify contents of this file. Note that for short palindromes the number of possible numbers is rather small, but we ignore this sampling collision problem for the purpose of this assignment.

<h2>1.2           Vanilla RNN in PyTorch</h2>

The vanilla RNN is formalized as follows. Given a sequence of input vectors x<sup>(<em>t</em>) </sup>for <em>t </em> 1, . . . <em>T</em>, the network computes a sequence of hidden states h<sup>(<em>t</em>) </sup>and a sequence of output vectors p<sup>(<em>t</em>) </sup>using the following equations for timesteps <em>t </em> 1, . . . , <em>T</em>:

h(<em>t</em>)  tanh(W<em>hx</em>x(<em>t</em>) + W<em>hh</em>h(<em>t</em>−1) + b<em>h</em>)                                                    (1)

p(<em>t</em>)  W<em><sub>ph</sub></em>h(<em>t</em>) + b<em><sub>p                                                                                                                                  </sub></em>(2)

As you can see, there are several trainable weight matrices and bias vectors. W<em><sub>hx </sub></em>denotes the input-to-hidden weight matrix, W<em><sub>hh </sub></em>is the hidden-to-hidden (or recurrent) weight matrix, W<em><sub>ph </sub></em>represents the hidden-to-output weight matrix and the b<em><sub>h </sub></em>and b<em><sub>p </sub></em>vectors denote the biases. For the first timestep <em>t </em> 1, the expression h<sup>(<em>t</em>−</sup><sup>1) </sup> h<sup>(</sup><sup>0) </sup>is replaced with a special vector h<em><sub>init </sub></em>that is commonly initialized to a vector of zeros. The output value p<sup>(<em>t</em>) </sup>depends on the state of the hidden layer h<sup>(<em>t</em>) </sup>which in its turn depends on all previous state of the hidden layer. Therefore, a recurrent neural network can be seen as a (deep) feed-forward network with shared weights.

To optimize the trainable weights, the gradients of the RNN are computed via backpropagation through time (BPTT). The goal is to calculate the gradients of the loss L with respect to the model parameters W<em><sub>hx</sub></em>, W<em><sub>hh </sub></em>and W<em><sub>ph </sub></em>(biases omitted). Similar to training a feed-forward network, the weights and biases are updated using SGD or one of its variants. Different from feed-forward networks, recurrent networks can give output logits yˆ<sup>(<em>t</em>) </sup>at every timestep. In this assignment the outputs will be given by the softmax function, <em>i.e. </em>yˆ<sup>(<em>t</em>) </sup> softmax(p<sup>(<em>t</em>)</sup>). For the task of predicting the final palindrome number, we compute the standard cross-entropy loss <em>only </em>over the last timestep:

<em>K</em>

L  −<sup>X</sup>y<em><sub>k </sub></em>logy<sup>ˆ</sup><em><sub>k                                                                                                          </sub></em>(3)

<em>k</em>1

Where <em>k </em>runs over the number of classes (<em>K </em> 10 because we have ten digits). In this expression, y denotes a one-hot vector of length <em>K </em>containing true labels.

<h3>Question 1.1 (10 points)</h3>

Recurrent neural networks can be trained using backpropagation through time. Similar to feed-forward networks, the goal is to compute the gradients of the loss w.r.t. W<em><sub>ph</sub></em>, W<em><sub>hh </sub></em>and W<em><sub>hx</sub></em>. Write down an expression for the gradient <u><sup>∂</sup></u><sub>∂</sub><u><sup>L</sup></u><sub>W</sub><sup>(</sup><em><sub>ph</sub></em><em><sup>T</sup></em><sup>) </sup>in terms of the variables that appear in Equations 1 and 2.

Do the same for <u><sup>∂</sup></u><sub>∂</sub><u><sup>L</sup></u><sub>W</sub><sup>(</sup><em><sub>hh</sub></em><em><sup>T</sup></em><sup>) </sup>. What difference do you observe in temporal dependence of the two gradients. Study the latter gradient and explain what problems might occur when training this recurrent network for a large number of timesteps.

<h3>Question 1.2 (10 points)</h3>

Implement the vanilla recurrent neural network as specified by the equations above in the file vanilla_rnn.py. For the <em>forward </em>pass you will need Python’s for-loop to step through time. You need to initialize the variables and matrix multiplications yourself without using high-level PyTorch building blocks. The weights and biases can be initialized using torch.nn.Parameter. The <em>backward </em>pass does not need to be implemented by hand, instead you can rely on automatic differentiation and use the RMSProp optimizer for tuning the weights. We have prepared boilerplate code in part1/train.py which you should use for implementing the optimization procedure.

<h3>Question 1.3 (5 points)</h3>

As the recurrent network is implemented, you are ready to experiment with the memorization capability of the vanilla RNN. Given a palindrome of length <em>T</em>, use the first <em>T </em>− 1 digits as input and make the network predict the last digit. The network is <em>successful </em>if it correctly predicts the last digit and thus was capable of memorizing a small amount of information for <em>T </em>timesteps.

Start with short palindromes (<em>T </em> 5), train the network until convergence and record the accuracy. Repeat this by gradually increasing the sequence length and create a plot that shows the accuracy versus palindrome length. As a sanity check, make sure that you obtain a near-perfect accuracy for <em>T </em> 5 with the default parameters provided in part1/train.py. To plot your results, evaluate your trained model for each palindrome length on a large enough separate set of palindromes, and repeat the experiment with different seeds to display stable results.

<h3>Question 1.4 (5 points)</h3>

To improve optimization of neural networks, many variants of stochastic gradient descent have been proposed. Two popular optimizers are RMSProp and Adam and/or the use of momentum. In practice, these methods are able to converge faster and obtain better local minima. In your own words, write down the benefits of such methods in comparison to vanilla stochastic gradient descent. Your answer needs to touch upon the concepts of momentum and adaptive learning rate.

Figure 1. A graphical representation of LSTM memory cells (Zaremba <em>et al. </em>(ICLR, 2015))

<h2>1.3           Long-Short Term Network (LSTM) in PyTorch</h2>

As you have observed after implementing the previous questions, training a vanilla RNN for remembering its inputs for an increasing number of timesteps is difficult. The problem is that the influence of a given input on the hidden layer (and therefore on the output layer), either decays or blows up exponentially as it unrolls the network. In practice, the <em>vanishing gradient problem </em>is the main shortcoming of vanilla RNNs. As a result, training a vanilla RNNs to consistently learn tasks containing delays of more than ∼ 10 timesteps between relevant input and target is difficult. To overcome this problem, many different RNN architectures have been suggested. The most widely used variant is the Long-Short

Term Network (LSTMs). An LSTM (Figure 1) introduces a number of gating mechanisms to improve gradient flow for better training. Before continuing, please read the following blogpost: <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding LSTM Networks</a><a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">.</a>

In this assignment we will use the following LSTM definition:

<table width="368">

 <tbody>

  <tr>

   <td width="344">g(<em>t</em>)  tanh(W<em>gx</em>x(<em>t</em>) + W<em>gh</em>h(<em>t</em>−1) + b<em>g</em>)</td>

   <td width="24">(4)</td>

  </tr>

  <tr>

   <td width="344">i(<em>t</em>) <sup> </sup>σ(W<em><sub>ix</sub></em>x(<em>t</em>) <sup>+ W</sup><em><sub>ih</sub></em>h(<em>t</em>−1) <sup>+ </sup>b<em><sub>i</sub></em>)</td>

   <td width="24">(5)</td>

  </tr>

  <tr>

   <td width="344">f(<em>t</em>)  σ(W<em>f x</em>x(<em>t</em>) + W<em>f h</em>h(<em>t</em>−1) + b<em>f </em>)</td>

   <td width="24">(6)</td>

  </tr>

  <tr>

   <td width="344">o(<em>t</em><sup>) </sup> σ(W<em><sub>ox</sub></em>x(<em>t</em><sup>) </sup>+ <sup>W</sup><em><sub>oh</sub></em>h(<em>t</em>−1) + b<em><sub>o</sub></em>)</td>

   <td width="24">(7)</td>

  </tr>

  <tr>

   <td width="344">c(<em>t</em>) <sup> </sup>g(<em>t</em>) <sup> </sup>i(<em>t</em>) <sup>+ </sup>c(<em>t</em>−1) <sup> </sup>f(<em>t</em>)</td>

   <td width="24">(8)</td>

  </tr>

  <tr>

   <td width="344">h(<em>t</em>)  tanh(c(<em>t</em>))  o(<em>t</em>)</td>

   <td width="24">(9)</td>

  </tr>

  <tr>

   <td width="344">p(<em>t</em>)  <sup>W</sup><em><sub>ph</sub></em>h(<em>t</em>) + b<em><sub>p</sub></em></td>

   <td width="24">(10)</td>

  </tr>

  <tr>

   <td width="344">yˆ<sup>(<em>t</em>) </sup> softmax(p<sup>(<em>t</em>)</sup>).</td>

   <td width="24">(11)</td>

  </tr>

 </tbody>

</table>

In these equations  is element-wise multiplication and σ(·) is the sigmoid function. The first six equations are the LSTM’s core part whereas the last two equations are just the linear output mapping. Note that the LSTM has more weight matrices than the vanilla RNN. As the forward pass of the LSTM is relatively intricate, writing down the correct gradients for the LSTM would involve a lot of derivatives. Fortunately, LSTMs can easily be implemented in PyTorch and automatic differentiation takes care of the derivatives.

<h3>Question 1.5 (5 points)</h3>

<ul>

 <li>(3 points) The LSTM extends the vanilla RNN cell by adding four gating mechanisms. Those gating mechanisms are crucial for successfully training recurrent neural networks. The LSTM has an <em>input modulation gate </em>g<sup>(<em>t</em>)</sup>, <em>input gate </em>i<sup>(<em>t</em>)</sup>, <em>forget gate </em>f<sup>(<em>t</em>) </sup>and <em>output gate </em>o<sup>(<em>t</em>)</sup>. For each of these gates, write down a brief explanation of their purpose; explicitly discuss the non-linearity they use and motivate why this is a good choice.</li>

 <li>(2 points) Given the LSTM cell as defined by the equations above and an input sample x ∈ R<em><sup>T</sup></em><sup>×<em>d </em></sup>where <em>T </em>denotes the sequence length and <em>d </em>is the feature dimensionality. Let <em>n </em>denote the number of units in the LSTM and <em>m </em>represents the batch size. Write down the formula for the <em>total number </em>of trainable parameters in the <em>LSTM cell </em>as defined above.</li>

</ul>

<h3>Question 1.6 (10 points)</h3>

Implement the LSTM network as specified by the equations above in the file lstm.py. Just like for the Vanilla RNN, you are required to implement the model without any high-level PyTorch functions. You do not need to implement the <em>backward </em>pass yourself, but instead you can rely on automatic differentiation and use the RMSProp optimizer for tuning the weights. For the optimization part we have prepared the code in train.py.

Using the palindromes as input, perform the same experiment you have done in <em>Question 1.3</em>. Train the network until convergence. You might need to adjust the learning rate when increasing the sequence length. The initial parameters in the prepared code provide a starting point. Again, create a plot of your results by gradually increasing the sequence length. Write down a comparison with the vanilla RNN and think of reasons for the different behavior. As a sanity check, your LSTM should obtain near-perfect accuracy for <em>T </em> 5.

We have now implemented the vanilla RNN and LSTM networks and have compared the difference in their performance at the task of palindrome prediction. We will now study the difference of their <em>temporal dependence of the gradients </em>as discussed in Question 1.1 for both the variants of the RNNs in more detail.

<h3>Question 1.7 (5 points)</h3>

Modify your implementations of RNN and LSTM to obtain the gradients between time steps of the sequence, namely <em><sub>dh</sub><u><sup>dL</sup></u></em><em><sub>t </sub></em>. You do not have to train a network for this task. Take as input a palindrome of length 100 and predict the number at the final time step of the sequence. Note that the gradients over time steps does not imply the gradients of the RNN/LSTM cell blocks, but the message that gets passed between time-steps (i.e, the hidden state). Plot the gradient magnitudes for both the variants over different time steps and explain your results. Do the results correlate with the findings in Question 1.6? What results will you expect if we actually train the network instead for this task? Submit your implementation as a separate file called grads_over_time.py.

<em>Hint: In PyTorch you can declare variables with requires_grad = True to add them to the computation graph and get their gradients when back-propagating the loss. Use this to extract gradients from the hidden state between time steps.</em>

<h1>2           Recurrent Nets as Generative Model     (Total: 30+5 points)</h1>

In this assignment you will build an LSTM for generation of text. By training an LSTM to predict the next character in a sentence, the network will learn local structure in text. You will train a two-layer LSTM on sentences from a book and use the model to generate new next. Before starting, we recommend reading the blog post <a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable </a><a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/">Effectiveness of Recurrent Neural Networks</a><a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/">.</a>

Given a training sequence x  (<em>x</em><sup>1</sup>, . . . , <em>x<sup>T</sup></em>), a recurrent neural network can use its output vectors p  (<em>p</em><sup>1</sup>, . . . , <em>p<sup>T</sup></em>) to obtain a sequence of predictions yˆ<sup>(<em>t</em>)</sup>. In the first part of this assignment you have used the recurrent network as <em>sequence-to-one </em>mapping, here we use the recurrent network as <em>sequence-to-sequence </em>mapping. The total cross-entropy loss can be computed by averaging over all timesteps using the target labels y<sup>(<em>t</em>)</sup>.

<em>K</em>

L(<em>t</em>)  −Xy<em>k</em>(<em>t</em>) logyˆ<em>k</em>(<em>t</em>)                                                                                               (12)

<em>k</em>1

L  1 X L(<em>t</em>)                                                                                                              (13)

<em>T</em>

<em>t</em>

Again, <em>k </em>runs over the number of classes (vocabulary size). In this expression, y denotes a one-hot vector of length <em>K </em>containing true labels. Using this sequential loss, you can train a recurrent network to make a prediction at every timestep. The LSTM can be used to generate text, character by character that will look similar to the original text. Just like multi-layer perceptrons, <a href="https://i.imgur.com/J3DwxSF.png">LSTM cells can be stacked</a> to create deeper layers for increased expressiveness. Each recurrent layer can be unrolled in time.

For training you can use a large text corpus such as publicly available books. We provide a number of books in the assets directory. However, you are also free to download other books, we recommend <a href="https://www.gutenberg.org/browse/languages/en">Project Gutenberg</a> as good source. Make sure you download the books in plain text (.txt) for easy file reading in Python. We provide the TextDataset class for loading the text corpus and drawing batches of example sentences from the text.

The files train_text.py and model.py provide a starting point for your implementation. The sequence length specifies the length of training sentences which also limits the number of timesteps for backpropagation in time. When setting the sequence length to 30 steps, the gradient signal will never backpropagate more than 30 timesteps. As a result, the network cannot learn text dependencies longer than this number of characters.

<h2>Question 2.1 (30 points)</h2>

We recommend reading <a href="https://pytorch.org/docs/stable/nn.html#recurrent-layers">PyTorch’s documentation on RNNs</a> before you start. Study the code and its outputs for part2/dataset.py to sample sentences from the book to train with. Also, have a look at the parameters defined in part2/train.py and implement the corresponding PyTorch code to make the features work. We obtained good results with the default parameters as specified, but you may need to tune them depending on your own implementation.

<ul>

 <li>(10 points) Implement a <em>two</em>-layer LSTM network to predict the next character in a sentence by training on sentences from a book. Train the model on sentences of length <em>T </em>30 from your book of choice. Define the total loss as average of cross-entropy loss over all timesteps (Equation 13). Plot the model’s loss and accuracy during training, and report all the relevant hyperparameters that you used and shortly explain why you used them.</li>

 <li>(10 points) Make the network generate new sentences of length <em>T </em>30 now and then by randomly setting the first character of the sentence. Report 5 text samples generated by the network over different stages of training. Carefully study the text generated by your network. What changes do you observe when the training process evolves? For your dataset, some patterns might be better visible when generating sentences longer than 30 characters. Discuss the difference in the quality of sentences generated (e.g. coherency) for a sequence length of less and more than 30 characters.</li>

 <li>(10 points) Your current implementation uses <em>greedy sampling</em>: the next character is always chosen by selecting the one with the highest probability. On the complete opposite, we could also perform <em>random sampling</em>: this will result in a high diversity of sequences but they will be meaningless. However, we can interpolate between these two extreme cases by using a <em>temperature </em>parameter τ in the softmax:</li>

</ul>

softmax(<em>x</em>˜) Pexp(τ<em>x</em>˜)

<em><sub>i </sub></em>exp(τ<em>x</em>˜<em><sub>i</sub></em>)

(for details, see <a href="https://www.deeplearningbook.org/">Goodfellow</a> <a href="https://www.deeplearningbook.org/"><em>et al.</em></a><a href="https://www.deeplearningbook.org/">;</a> Section 17.5.1).

<ul>

 <li>Explain the effect of the temperature parameter τ on the sampling process.</li>

 <li>Extend your current model by adding the temperature parameter τ to balance the sampling strategy between fully-greedy and fully-random.</li>

 <li>Report generated sentences for temperature values τ ∈ {0.5, 1.0, 2.0}. What do you observe for different temperatures? What are the differences with respect to the sentences obtained by greedy sampling?</li>

</ul>

<em>Note that using one single dataset is sufficient to get full points. However, we encourage you to experiment using different datasets to gain more insight. We suggest to start with relatively small (and possibly with simple language) datasets, such as Grim’s fairy tales, so that you can train the model on your laptop easily. If the network needs training for some hours until convergence, it is advised to run training on the SurfSara cluster. Also, you might want to save the model checkpoints now and then so you can resume training later or generate new text using the trained model.</em>

<h2>Bonus Question 2.2 (2 points)</h2>

It could be fun to make your network finish some sentences that are related to the book that your are training on. For example, if you are training on Grimm’s Fairytales, you could make the network finish the sentence “<em>Sleeping beauty is …</em>”. Be creative and test the capabilities of your model. What do you notice? Discuss what you see.

<h2>Bonus Question 2.3 (3 points)</h2>

There is also anothermethod that could be used forsampling: Beam Search. Shortly describe how it works, and mention when it is particularly useful. Implement it (you can choose the beam size), and discuss the results.

<h1>3           Graph Neural Networks      (Total: 20 points)</h1>

<h2>3.1           GCN Forward Layer</h2>

Graph convolutional neural networks are widely known architectures used to work with graph-structured data, and a particular version (GCN, or Graph Convolutional Network) was firstly introduced in <a href="https://arxiv.org/pdf/1609.02907.pdf">https://arxiv.org/pdf/1609.02907.pdf</a><a href="https://arxiv.org/pdf/1609.02907.pdf">.</a> Consider Eq. 14, describing the propagation rule for a layer in the Graph Convolutional Network architecture to answer the following questions.

<table width="515">

 <tbody>

  <tr>

   <td width="176">Where <em>A</em><sup>ˆ </sup>is obtained by:</td>

   <td width="255">          <em>H</em>(<em>l</em>+1)  σ(<em>AH</em>ˆ      (<em>l</em>)<em>W</em>(<em>l</em>))</td>

   <td width="84">(14)</td>

  </tr>

  <tr>

   <td width="176"> </td>

   <td width="255"><em>A</em>ˆ  <em>D</em>˜−<u>1</u>2<em>A</em>˜<em>D</em>˜−<u>1</u>2</td>

   <td width="84">(15)</td>

  </tr>

  <tr>

   <td width="176"> </td>

   <td width="255"><em>A</em><sup>˜ </sup> <em>A </em>+ <em>I<sub>N</sub></em></td>

   <td width="84">(16)</td>

  </tr>

  <tr>

   <td width="176"> </td>

   <td width="255"><em>D</em>˜<em>ii </em>X<em>A</em>˜<em>ij</em></td>

   <td width="84">(17)</td>

  </tr>

 </tbody>

</table>

<em>j</em>

In the equations above, <em>H</em><sup>(<em>l</em>) </sup>is the <em>N </em>× <em>d </em>matrix of activations in the <em>l</em>-th layer, <em>A </em>is the <em>N </em>× <em>N </em>adjacency matrix of the undirected graph, <em>I<sub>N </sub></em>is an identity matrix of size <em>N</em>, and <em>D</em><sup>˜ </sup>is a diagonal matrix used for normalization (you don’t need to care about this normalization step, instead, you should focus on discussing Eq. 14). <em>A</em><sup>˜ </sup>is the adjacency matrix considering self-connections, <em>N </em>is the number of nodes in the graph, <em>d </em>is the dimension of the feature vector for each node. The adjacency matrix A is usually obtained from data (either by direct vertex attribute or by indirect training). W is a learnable <em>d</em><sup>(<em>l</em>) </sup><sup>× </sup><em>d</em><sup>(<em>l</em>+</sup><sup>1) </sup>matrix utilized to change the dimension of feature per vertex during the propagation over the graph.

Figure 2. Example graph for Question 3.2. The graph consists of 6 nodes (V  {<em>A</em>, <em>B</em>, <em>C</em>, <em>D</em>, <em>E</em>, <em>F</em>}), where connections between them represent the undirected edges of the graph.

<h3>Question 3.2 (4 points)</h3>

Consider the following graph in Figure 3 for the following questions:

<ul>

 <li>(2 points) Give the adjacency matrix <em>A</em><sup>˜ </sup>for the graph as specified in Equation 16.</li>

 <li>(2 points) How many updates (as defined in Equation 14) will it take to forward the information from node C to node E?</li>

</ul>

<h2>3.2           Applications of GNNs</h2>

Models based on Graph Neural Networks can be efficiently applied for a variety of realworld applications where data can be described in form of graphs.

<h3>Question 3.3 (2 points)</h3>

Take a look at the publicly available literature and name a few real-world applications in which GNNs could be applied.

<h2>3.3           Comparing and Combining GNNs and RNNs</h2>

Structured data can often be represented in different ways. For example, images can be represented as graphs, where neighboring pixels are nodes connected by edges in a pixel graph. Also sentences can be described as graphs (in particular, trees) if we consider their Dependency Tree representation. On the other direction, graphs can be represented as sequences too, for example through DFS or BFS ordering of nodes. Based on this idea, answer the following open-answer questions (notice that there is not a unique, correct answer in this case, so try to discuss your ideas, supporting them with what you believe are valid motivations).

<h3>Question 3.4 (8 points)</h3>

<ul>

 <li>(6 points) Consider using RNN-based models to work with sequence representations, and GNNs to work with graph representations of some data. Discuss what the benefits are of choosing either one of the two approaches in different situations. For example, in what tasks (or datasets) would you see one of the two outperform the other? What representation is more expressive for what kind of data?</li>

 <li>(2 points) Think about how GNNs and RNNs models could be used in a combined model, and for what tasks this model could be used. Feel free to mention examples from literature to answer this question.</li>

</ul>

<h1>Report</h1>

We expect each student to write a small report about recurrent neural networks with explicitly answering the questions in this assignment. Please clearly mark each answer by a heading indicating the question number. Again, use the NIPS L<sup>A</sup>TEX template as was provided for the first assignment.

<h1>Deliverables</h1>

Create ZIP archive containing your report and all Python code. Please preserve the directory structure as provided in the Github repository for this assignment. Give the ZIP file the following name: lastname_assignment2.zip where you insert your lastname. Please submit your deliverable through Canvas. We cannot guarantee a grade for the assignment if the deliverables are not handed in according to these instructions.