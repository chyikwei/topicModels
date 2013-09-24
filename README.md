In current Mallet package, it only contains two topic Models--LDA and Hierachical LDA.<br />
So I tried to implement some useful topic modeling method on it:<br />
 * Hierarchical Dirichlet Process
 * inference part for hLDA

Usage:

1. This is an extension for Mallet, so you need to have Mallet's source code first.<br  />
2. put <code>HDP.java</code>,<code>HDPInferencer.java</code> and <code>HierarchicalLDAInferencer.java</code> in <code>src/cc/mallet/topics</code> folder.
3. If you are going to run HDP, make sure you have <code>knowceans</code> package.<br  />
4. run <code>HDPTest.java</code> or <code>hLDATest.java</code> will give you a demo for a small dataset in <code>data</code> folder.

References:
 * Mallet: http://mallet.cs.umass.edu/
 * knowceans: http://sourceforge.net/projects/knowceans/
 * HDP paper: http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf
 * HDP paper & source code: "Implementing the HDP with minimum code complexity" by Gregor Heinrich

--------------

Update History:

 >2012/10/01 version 0.1
 >- bug fix: print correct topic number in training
 >- add cross validation in HDP
 >- add inferencer class
 >- add preplexity calculation in inferencer
 
 >2012/09/29 Version 0.1
 >- bug fix: printed result are correct now
 >- bug fix: empty topic are caused by initial topic number > 0
 >- change initial topic assignment to uniform distribution and remove empty topics. 
 
 >2012/09/28 Version 0.1
 >- bug: Topic number and total word count not match in printed result
 >- bug: some topics are empty but not removed  
 
 >2012/09/27 Version 0.1
 >- main algorithm work. not finished all function, 
 >- bug: auto update hyper-parameter doesn't work well. Disable it for now.
