In current mallet package, it only contains two topic Models--LDA and Hierachical LDA.
I tried to implement a Hierarchical Dirichlet Process for this package.

If you are a Mallet user, you can put HDP.java in the "cc.mallet.topics" packages and try HDPTest.java for test.

The source code I got is for the paper "Implementing the HDP with minimum code complexity" by Gregor Heinrich
So you will also need the "knowceans" package.

You can find all you need in the following reference:

Mallet:
http://mallet.cs.umass.edu/

knowceans:
http://sourceforge.net/projects/knowceans/

HDP paper:
http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf