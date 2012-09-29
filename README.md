In current Mallet package, it only contains two topic Models--LDA and Hierachical LDA.<br />
So I tried to implement some useful topic modeling method on it:<br />
 * Hierarchical Dirichlet Process
 * inference part for hLDA

Note:

1. It a extension for Mallet, so you need to include them in Mallet's source code.<br  />
2. You will also need the "knowceans" package.<br  />
3. For HDP, The source code I got is for the paper "Implementing the HDP with minimum code complexity" by Gregor Heinrich<br /> 
4. ISI_Abstract_original.txt is used for testing.<br  />

Also, you can find all you need in the following reference:
 * Mallet: http://mallet.cs.umass.edu/
 * knowceans: http://sourceforge.net/projects/knowceans/
 * HDP paper: http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf

--------------

Update History:

>2012/09/28 version 0.1
>- primitive model. Inference part not finished yet.
