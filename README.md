1. Mallet Extension
-------------------
In Mallet package, it only contains two topic Models--LDA and Hierachical LDA. 
So I tried to implement some useful topic modeling methods on it.<br/>

__Model:__
 *  Hierarchical Dirichlet Process with Gibbs Sampling. (in `HDP` folder)
 *  Inference part for hLDA. (in `hLDA` folder)

__Usage:__ 

1. This is an extension for Mallet, so you need to have Mallet's source code first.
2. put `HDP.java`, `HDPInferencer.java` and `HierarchicalLDAInferencer.java` in `src/cc/mallet/topics` folder.
3. If you are going to run HDP, make sure you include `knowceans` package in your project.
4. run `HDPTest.java` or `hLDATest.java` will give you a demo for a small dataset in `data` folder.

**References:**

 * [Mallet](http://mallet.cs.umass.edu/)
 * [knowceans](http://sourceforge.net/projects/knowceans/)
 * [HDP paper](http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf)
 * "Implementing the HDP with minimum code complexity" by Gregor Heinrich


2. Scikit-learn Extension
-------------------------
**Note**:

  This extension is merged in scikit-learn 0.17 version. Please check [here](https://github.com/scikit-learn/scikit-  learn/blob/master/sklearn/decomposition/online_lda.py) for latest code.

**Model**:

 * online LDA with variational inference. (In `LDA` folder)


**Usage**:

  1. Make sure `numpy`, `scipy`, and `scikit-learn` are installed. 
  2. run `python test` in `lda` folder for unit test
  3. The onlineLDA model is in `lda.py`. 
  4. For a quick exmaple, run`python lda_example.py online` will fit a 10 topics model with 20 NewsGroup dataset. `online` means we use online update(or `partial_fit` method). Change `online` to `batch` will fit the model with batch update(or `fit` method).

__Reference:__
 * [Scikit-learn](http://scikit-learn.org)
 * [onlineLDA](http://www.cs.princeton.edu/~mdhoffma/code/onlineldavb.tar)
 * [online LDA paper](http://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf)

**Others**:

 * Another HDP implementation can be found it my [bnp](https://github.com/chyikwei/bnp) repository. It also follows scikit-learn API  and is optimized with cython.
 
