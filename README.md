# Multinomial Regression(MNR)
This repository contains the implementation of multinomial regression on iris dataset by considering backtracking method. Then it compute misclassification rate of MNR and finaly compare the effiency of gradient descent and stochastic gradient descent.\
{Steps to create MNR model}
\begin{itemize}
    \item \textbf{Step-0:} Initialize the weight matrix and bias values with zero or a small random values.
    \item \textbf{Step-1:} For each class of K we compute a linear combination of the input features and the weight vector of class K, which is for each training example compute a score for each class. For class K and input vector $\boldsymbol{x}_{(i)}$ we have:
    \\
    \\$s_{k}(\boldsymbol{x}_{(i)}) = \boldsymbol{w}_{k}^T \cdot \boldsymbol{x}_{(i)} + b_{k}$
    \\
    \\In this equation $\cdot$ is the dot product and $\boldsymbol{w}_{(k)}$ the weight vector of class K. So we can find and compute the s which is scores for all classes and training examples in parallel, using vectorization and broadcasting:
    \\
    \\$\boldsymbol{S} = \boldsymbol{X} \cdot \boldsymbol{W}^T + \boldsymbol{b} $
    \\

    \\which $\boldsymbol{X}$ is a matrix of data that has $n_{samples}$ and $n_{features}$ that holds all training examples, and $\boldsymbol{W}$ is a matrix in shape of $n_{classes}$ and $n_{features}$ that holds the weight vector for each class.
    \\
    \item \textbf{Step-2:} Apply the softmax  function for activation to transform the scores into probabilities. The probability that an input vector $\boldsymbol{x}_{(i)}$ belongs to class k is given by:
    \\
    \\$\hat{p}_k(\boldsymbol{x}_{(i)}) = \frac{\exp(s_{k}(\boldsymbol{x}_{(i)}))}{\sum_{j=1}^{K} \exp(s_{j}(\boldsymbol{x}_{(i)}))}$
    \\
    \\ In this step we have to perform all the formulas for all classes and our training examples when we using vectorization. We can see the class predicted by this model for $\boldsymbol{x}_{(i)}$  is then simply the class with the highest probability.
    \\
    \item \textbf{Step-3:} In this step we should calculate the cost over the whole training set. The result that we expected from this step is our model predict a high probability for our targeted class and the lowest probability for other classes. So it can use the cross entropy loss function:
    \\

    \\$h(\boldsymbol{W},b) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^{K} \Big[ y_k^{(i)} \log(\hat{p}_k^{(i)})\Big]$
    \\
    \\ Here we use one-hot encoding, because if we used numerical categories 1,2,3,... we would impute ordinal. So $y_k^{(i)}$ is 1 for the targeted class and for the other $\boldsymbol{x}^{(i)}$ for k classes  $y_k^{(i)}$ should be 0.
    \\
    \item \textbf{Step-4:} In this step we need to compute gradient of the cost function for each weight vector and bias: for class k we have:
    \\
    \\$ \nabla_{\boldsymbol{w}_k} h(\boldsymbol{W}, b) = \frac{1}{m}\sum_{i=1}^m\boldsymbol{x}_{(i)} \left[\hat{p}_k^{(i)}-y_k^{(i)}\right]$
    \\
    \\
    \item \textbf{Step-5:} Here we just need to update biases and weights for all the classes of k an $\eta$ is my learning rate or step length:
    \\
    \\$\boldsymbol{w}_k = \boldsymbol{w}_k - \eta \, \nabla_{\boldsymbol{w}_k} h$
    \\$b_k = b_k - \eta \, \nabla_{b_k} h$
\\
\end{itemize}
\section{Experimental Results}
\subsection{Decision Region for training MNR with No Regularization}
In this project I use Iris dataset: 
\begin{figure}[h!]
\centering
\includegraphics[width=0.3\textwidth]{irisset.png}
\caption{\label{fig:irisset}In this data-set we have 3 categories of iris flowers}
\end{figure}
\\For implementing the multinumial regression, I follow the above steps and train MNR with $\lambda=0$ without regularization and show the obtained decision regions in a figure below. 
\\To implement gradient descent with backtracking, the maximum learning rate is 300 and stopping the condition when $\frac{d_{ACE}}{d_w}<0.01$. In this dataset, MNR uses linear decision boundary. Because linear boundaries are good enough to give very good rate.
\begin{itemize}
    \item Training error for 120 samples is 0.02
\end{itemize}\
Click [here](https://github.com/NedaKeivan/Pattern-Recognition-MNR-/blob/main/MNR-complete-report.pdf) to go to the complete report of this ICP Project Repository.
