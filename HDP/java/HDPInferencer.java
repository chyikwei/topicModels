package cc.mallet.topics;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;

public class HDPInferencer {
	
	 private InstanceList testInstances;
	 private Alphabet alphabet;
	 private int K;
	 private double alpha;
	 private int numTypes;
	 private int numDocs;
	 private Random rand;
	 
	 private double[][] phi;
	 private int[][] nmk;
	 private int[][] z; //topic indicator
	 private int[] effectiveDocLength;
	 private int totalTokens=0;
	 public HDPInferencer(double alpha, int K, double[][] phi, Alphabet alphabet, Random rand){
		 this.alpha=alpha;
		 this.K=K;
		 this.phi=phi;
		 this.numTypes= phi[0].length;
		 this.alphabet = alphabet;
		 this.rand =rand;
	 }
	 
	 public void setInstance(InstanceList instances){
		 
		 this.testInstances = instances;
		 numDocs = testInstances.size();
		 effectiveDocLength = new int[numDocs];
		 //init
		 init();
	 }
	 
	 /*initialization*/
	 private void init(){
		 
		 nmk = new int[numDocs][K];
		 z = new int[numDocs][];
		 int type, token, seqLen;
		 FeatureSequence fs;
		 for(int m=0; m < numDocs; m++){
		     	 
			 fs =(FeatureSequence) testInstances.get(m).getData();
			 seqLen=fs.getLength();
			 effectiveDocLength[m]=seqLen;
			 totalTokens+=seqLen;
			 z[m]= new int[seqLen];
			 
			 for(token=0 ; token < seqLen ; token++){
				 
				 type = fs.getIndexAtPosition(token);
				 
				 //set unseen word to "-1"
			     if(type >= numTypes){
			    	 z[m][token]=(-1);
			    	 effectiveDocLength[m]--;
			    	 totalTokens--;
			     }
			     else{
			    	 int k = rand.nextInt(K);
					 z[m][token]=k;
					 nmk[m][k]++;
			     }
			 }//end for

		 }
	 }
	 
	 /*estimate*/
	 public void estimate(int iterations){
		 
		 for(int iter=0; iter<iterations ; iter++){
			 for(int m=0 ; m < numDocs ; m++){
				 updateDocs(m);
			 }
		 }
	 }
	 
	 private void updateDocs(int m){
		 
		 FeatureSequence fs = (FeatureSequence) testInstances.get(m).getData();
		 int seqLen = fs.getLength();
		 int type, token;
		 double sum;
		 double[] pp = new double[K];
		 
		 for(token=0 ; token < seqLen ; token++){
			 
			 //skip unseen word
			 if(z[m][token]<0){continue;}
			 
			 type = fs.getIndexAtPosition(token);
			 
			 //decrement
			 int k = z[m][token];
			 nmk[m][k]--;
			 
			 sum=0.0;
			 for(int kk=0 ; kk<K; kk++){
				 pp[kk]=(nmk[m][kk]+alpha)* phi[kk][type];
				 sum+=pp[kk];
			 }
			 
			 //sample
			 double u = rand.nextDouble();
			 u*=sum;
			 
			 sum=0.0;
			 
			 int kk=0;
			 for(; kk<K; kk++){
				 sum+=pp[kk];
				 
				 if(u <= sum) break;
			 }
			 
			 z[m][token]=kk;
			 nmk[m][kk]++;
		 }
	 }
	 
	 public double[] topicDistribution(int m){
		 
		 if(m >= numDocs){
			 throw new ArrayIndexOutOfBoundsException();
		 }
		 
		 double[] distr = new double[K];
		 double alphaSum = K * alpha;
		 int totalCount=0;
		 
		 for(int k=0; k<K ;k++){
			 totalCount+=nmk[m][k];
		 }
		 
		 for(int k=0 ; k<K ; k++){
			 
			 //word count only
			 //distr[k] = (double)nmk[m][k];
			 
			 //smoothed probability
			 distr[k] = (alpha + nmk[m][k])/(alphaSum + totalCount);
		 }
		 
		 return distr;
	 }

	 /*preplexity*/
	 public double getPreplexity(){
		 
		 //TODO
		 double preplexity=0.0;
		 double logLik=0.0;
		 double[][] theta = new double[numDocs][K];
		 
		 //calculate theta
		 for(int m=0 ; m<numDocs ; m++){
			 for(int k=0 ; k< K ; k++){
				 theta[m][k]= ((double)nmk[m][k]+alpha)/(effectiveDocLength[m]+K*alpha);
			 }
		 }
		 
		 //calculate LL
		 for(int m=0 ; m<numDocs ; m++){
			 
			 FeatureSequence fs = (FeatureSequence) testInstances.get(m).getData();
			 int seqLen = fs.getLength();
			 int type, token;
			 
			 for(token=0 ; token < seqLen ; token++){
				 
				 type = fs.getIndexAtPosition(token);
				 //only consider existed type
				 if(type < numTypes){
					 double sum =0.0;
					 for(int k=0 ; k<K ; k++){
						 sum += (theta[m][k]*phi[k][type]);
					 }//end k
					 logLik += Math.log(sum);
				 }//end if
			}//end token
		 }//end m
		 
		 preplexity =Math.exp( (-1)*logLik / totalTokens);
		 
		 return preplexity;
	 }
	 
}
