1. Mallet Extension
-------------------
In Mallet package, it only contains two topic Models--LDA and Hierachical LDA. 
So I tried to implement some useful topic modeling methods on it.<br/>

__Model:__
 *  Hierarchical Dirichlet Process with Gibbs Sampling. (in <code>HDP</code> folder)
 *  Inference part for hLDA. (in <code>hLDA</code> folder)

__Usage:__ 

1. This is an extension for Mallet, so you need to have Mallet's source code first.
2. put <code>HDP.java</code>,<code>HDPInferencer.java</code> and <code>HierarchicalLDAInferencer.java</code> in <code>src/cc/mallet/topics</code> folder.
3. If you are going to run HDP, make sure you include <code>knowceans</code> package in your project.
4. run <code>HDPTest.java</code> or <code>hLDATest.java</code> will give you a demo for a small dataset in <code>data</code> folder.

__References:__
 * Mallet: http://mallet.cs.umass.edu/
 * knowceans: http://sourceforge.net/projects/knowceans/
 * HDP paper: http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf
 * HDP paper & source code: "Implementing the HDP with minimum code complexity" by Gregor Heinrich


2. Scikit-learn Extension
-------------------------
__Note__:<br>
This extension is merged in scikit-learn 0.17 version.

__Model__: <br>
 * online LDA with variational EM. (In <code>LDA</code> folder)

__Usage__: <br>

1. Make sure `numpy`, `scipy`, and `scikit-learn` are installed. 
2. run `python test` in `lda` folder for unit test
3. The onlineLDA model is in <code>lda.py</code>. 
4. For a quick exmaple, run <code>python lda_example.py online</code> will fit a 10 topics model with 20 NewsGroup dataset. `online` means we use online update(or `partial_fit` method). Change `online` to `batch` will fit the model with batch update(or `fit` method).


__Reference:__
 * Scikit-learn: http://scikit-learn.org
 * onlineLDA: http://www.cs.princeton.edu/~mdhoffma/code/onlineldavb.tar
 * online LDA paper: http://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf
