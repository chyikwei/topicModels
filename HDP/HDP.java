/* Hierarchical Dirichlet Process for LDA
 * Version:0.1
 * 
 * Author: CHyi-Kwei Yau
 * Latest Update: 2012/09/27
 * 
 * HDP implementation on Mallet 
 * Algorithms & Code form "Implementing the HDP with minimum code complexity" by Gregor Heinrich
 * 
 * */

package cc.mallet.topics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.IDSorter;
import cc.mallet.types.InstanceList;

import org.knowceans.util.ArrayUtils;
import org.knowceans.util.DirichletEstimation;
import org.knowceans.util.IndexQuickSort;
import org.knowceans.util.Samplers;
import org.knowceans.util.Vectors;

public class HDP {
	
	 InstanceList instances;
	 InstanceList testIstances;
	 
	 private int currentIter;
	 
	 private Random rand;
	 /*root base measure*/
	 private double beta;
	 private double gamma;
	 
	 /*2nd level Dirichlet Process parameter*/
	 private double alpha;//precision
	 private ArrayList<Double> tau; //mean
	 
	 /*hyper parameter*/
	 private double a_alpha;
	 private double a_beta;
	 private double a_gamma;
	 private double b_alpha;
	 private double b_beta;
	 private double b_gamma;
	 
	 /*Training Instance*/
	 private int numDocuments; //num of doc
	 private int numTypes; //num of vocabulary
	 private int totalWord=0;
	 private int topWordNum = 6; //display number of top word
	 
	 /*algorithm parameter*/
	 /*for training*/
	 private List<Integer>[] nmk;
	 private List<int[]> nkt;
	 private List<Integer> nk;
	 private double[][] phi;
	 private int[][] z; //topic indicator
	 private double[] pp; //topic distribution
	 private final int ppStep =10;
	 
	 private double tables;
	 
	 //number of samples for parameter samplers
	 private int parameterSampleNum=10;
	 
	 private int K; //initial topic number
	 private List<Integer> kgaps;
	 private List<Integer> kactive;
	 
	 /*non-empty parameter*/
	 private boolean initialized = false;
	 private boolean fixedK = false;
	 private boolean fixedHyper = true;
	 
	 /*for inference*/
	 
	 
	 /*set parameter*/
	 public HDP(){
		 this(1.0, 0.2, 1.0, 5);
	 }
	 
	public HDP(double alpha, double beta, double gamma, int initialK){
		 this.alpha = alpha;
		 this.beta = beta;
		 this.gamma = gamma;
		 
		 this.a_alpha=1.0;
		 this.a_beta =1.0;
		 this.a_gamma=5.0;
		 this.b_alpha=1.0;
		 this.b_beta=1.0;
		 this.b_gamma=5.0;
	
		 this.K = initialK;	 
		 this.currentIter=0;
		 
		 rand = new Random();
		 
		 if(gamma== 0){
			 this.fixedK=true;
		 }
	 }
	 
	 /*initialization*/
	 public void initialize(InstanceList instances){
		 
		 if (! (instances.get(0).getData() instanceof FeatureSequence)) {
				throw new IllegalArgumentException("Input must be a FeatureSequence, using the --feature-sequence option when impoting data, for example");
			}
		 
		 this.instances=instances;
		 numDocuments=instances.size();
		 numTypes=instances.getDataAlphabet().size();
		 
		 FeatureSequence fs;
		 for(int i=0;i<numDocuments ; i++){
			 fs =(FeatureSequence) instances.get(i).getData();
			 totalWord+=fs.getLength();
		 }
		 
		 /*initialize algorithm parameter*/
		 init();
		 
	 }
	 
	 /*initialize algorithm parameter*/
	 private void init(){
		 
		 nmk = new ArrayList[numDocuments];
		 nkt = new ArrayList<int[]>();
		 nk = new ArrayList<Integer>();
		 z = new int[numDocuments][];
		 
		 FeatureSequence fs;
		 for(int m=0; m < numDocuments; m++){
			 nmk[m] = new ArrayList<Integer>();
			 
			 for(int k=0; k<K ; k++){
				 nmk[m].add(0);
			 }
			 fs =(FeatureSequence) instances.get(m).getData();
			 z[m]= new int[fs.getLength()];
		 }
		 
		 /*topic index*/
		 kgaps = new ArrayList<Integer>();
		 kactive = new ArrayList<Integer>();
		 tau = new ArrayList<Double>();
		 
		 for(int k=0 ; k<K ; k++){
			 kactive.add(k);
			 nkt.add(new int[numTypes]);
			 nk.add(0);
			 tau.add(1.0/K);
		 }
		 //add one more topic
		 tau.add(1.0/K);
		 pp = new double[(K+ppStep)];
		 
		 //estimate once
		 estimate(1, false);
		 
		 if(!fixedK){
			 updateTau();
		 }
		 
		 initialized=true;
	 }
	 
	

	/*estimate*/
	 public void estimate(int iterations, boolean printResult){
		 
		 for(int iter=0; iter < iterations ; iter++){
			 
			 for(int i=0 ; i < numDocuments ; i++){		 
				 updateDocs(i);
			 }
			 
			 if(!fixedK){
				 updateTau();
			 }
			 
			 if(iter > 10 && !fixedHyper){
				 updateHyper();
			 }
				 
			 //print current status
			 if( iter !=0 && (iter % 50) == 0){ 
				 System.out.println("Iteration=" + iter +"\tTopic=" + (K-1) );
				 
				 if(!fixedHyper){
					 printParameter();
				 }
			 }
		 }//end iter
		 
		 //accumulate iterations
		 currentIter+=iterations;
		 
		 if(printResult){
			 //print a summary of estimation
			 System.out.println();
			 printParameter();
			 System.out.println();
			 printTopWord(topWordNum);
		}
		 
	 }
	 
	 private void updateDocs(int m) {
		 
		 FeatureSequence fs = (FeatureSequence) instances.get(m).getData();
		 int seqLen = fs.getLength();
		 int type, token;
		 double sum;
		 
		 for(token=0 ; token < seqLen ; token++){
			 
			 type = fs.getIndexAtPosition(token);
			 int k, kold = -1;
			 
			 if(initialized){
				 //get old topic
				 k=z[m][token];
				 //decrement
				 nmk[m].set(k, nmk[m].get(k)-1);
				 nkt.get(k)[type]--;
				 nk.set(k, nk.get(k)-1);
				 kold=k;
			 }
			 
			 sum=0.0;
			 for(int kk=0; kk<K ; kk++){
				 k=kactive.get(kk);
				 pp[kk] = (nmk[m].get(k) + alpha*tau.get(k))*
						 (nkt.get(k)[type]+beta) / (nk.get(k)+ numTypes*beta);
				 
				 sum+=pp[kk];
			 }
			 if(!fixedK){
				 pp[K] = alpha * tau.get(K) / numTypes;
				 sum+=pp[K];
			 }
			 
			 //sampling
			 double u = rand.nextDouble();
			 u *= sum;
			 sum=0.0;
			 int kk;
			 
			 for(kk=0 ; kk<=K ; kk++){
				 sum+=pp[kk];
				 if(u <= sum) break;
			 }
			 
			 //check kk is old or new topic
			 if(kk < K) //in old topic
			 { 
				 k = kactive.get(kk);
				 z[m][token]=k;
				 //add z back
				 nmk[m].set(k, nmk[m].get(k)+1);
				 nkt.get(k)[type]++;
				 nk.set(k, nk.get(k)+1);
			 }
			 else //add new topic
			 { 
				 assert(!fixedK);
				 z[m][token]=addTopic(m,type);
				 updateTau();
				 //System.out.println("add K="+K);
			 }
			 
			 //disable empty topic
			 if(initialized && nk.get(kold)==0){
				 kactive.remove((Integer)kold);
				 kgaps.add(kold);
				 assert(Vectors.sum(nkt.get(kold))==0 && 
						 nk.get(kold)==0 && nmk[m].get(kold)==0);
				 K--;
				 //System.out.println("remove K="+K);
				 
				 updateTau();
			 }
		 }
	 }

	private void updateHyper(){
		for(int r=0 ; r<parameterSampleNum ; r++){
			
			//check paper!
			//gamma: root level
			double eta = Samplers.randBeta(gamma+1 , tables);
			double bloge =b_gamma - Math.log(eta);
			double pie = (1.0 / (tables * bloge/ (gamma+K-1)));
			int u = Samplers.randBernoulli(pie);
			gamma = Samplers.randGamma(a_gamma+K-1+u, 1.0/bloge);
			
			//alpha: document level
			double qs = 0.0;
			double qw = 0.0;
			
			FeatureSequence fs;
			int seqLen;
			for(int m=0 ; m< numDocuments ; m++){
				fs = (FeatureSequence) instances.get(m).getData();
				seqLen = fs.getLength();
				qs += Samplers.randBernoulli(seqLen/(seqLen+alpha));
				qw += Math.log( Samplers.randBeta(alpha+1, seqLen));
			}
			alpha = Samplers.randGamma(a_alpha+ tables -qs, 1.0/(b_alpha-qw));
		}
		
		
		//estimate beta
		//convert nk & akt to array
		int[] ak = (int[]) ArrayUtils.asPrimitiveArray(nk);
		int[][] akt =new int[K][numTypes];
		for(int k=0; k<K; k++){
			akt[k] = nkt.get(k);
		}
		beta = DirichletEstimation.estimateAlphaMap(akt, ak, beta, a_beta, b_beta);
		
	 }
	 
	private void updateTau() {
		double[] mk = new double[K+1];
		
		for(int kk=0 ; kk<K ; kk++){
			int k = kactive.get(kk);
			
			for(int m=0 ; m < numDocuments ; m++){
				
				if(nmk[m].get(k) > 1){
					//sample number of tables
					mk[kk] +=Samplers.randAntoniak(alpha * tau.get(k), 
							nmk[m].get(k));
				}
				else //nmk[m].get(k) = 0 or 1
				{   
					mk[kk] +=nmk[m].get(k);
				}
				
			}
		}// end outter for loop
		
		//get number of tables
		tables = Vectors.sum(mk);
		mk[K] = gamma;
		
		double[] tt =Samplers.randDir(mk);
		
		for(int kk=0 ; kk < K ; kk++){
			
			int k=kactive.get(kk);
			tau.set(k, tt[kk]);
		}
		tau.set(K, tt[K]);
	}
	
	private int addTopic(int m, int type) {
		int k;
		if(kgaps.size()>0){ //reuse gaps
			k=kgaps.remove(kgaps.size()-1);
			kactive.add(k);
			nmk[m].set(k,1);
			nkt.get(k)[type]=1;
			nk.set(k,1);
		}
		else{
			k=K;
			
			for(int i=0 ; i< numDocuments ; i++){
				nmk[i].add(0);
			}
			
			kactive.add(K);
			nmk[m].set(K,1);
			nkt.add(new int[numTypes]);
			nk.add(1);
			tau.add(0.0);
		}
		//add topic number 
		K++;
		if(pp.length <=K){
			pp = new double[K + ppStep];
		}
		return k;
	}
	
	 /*print*/
	 public void printTopWord(int numWords){
		 
		//trimTopics();
		
		//int[] ak = (int[]) ArrayUtils.asPrimitiveArray(nk);
		//Arrays.sort(ak);
		
		int wordCount=0;
		
		for(int k=0 ; k<(K-1); k++){
			
			int kk=kactive.get(k);
			
			if(nk.get(kk)!=0){
				
				//check word count
				wordCount+=nk.get(kk);
				
				IDSorter[] sortedTypes = new IDSorter[numTypes];
				//sort word in topic k
				for (int type=0; type < numTypes; type++) {
					sortedTypes[type] = new IDSorter(type, nkt.get(k)[type]);
				}
				
				Arrays.sort(sortedTypes);
			
				Alphabet alphabet = instances.getDataAlphabet();
				StringBuffer out = new StringBuffer();
				out.append("topic"+kk + ": ");
				out.append("word:"+ nk.get(kk) + ", ");
				double prop = (double)nk.get(kk)/totalWord;
				out.append(String.format("prop:%2.4f, ", prop));
			
				for (int i=0; i<numWords; i++) {
					out.append(alphabet.lookupObject(sortedTypes[i].getID()) + " ");
				}
				System.out.println(out);
			}
			else{
				System.out.println("Topic"+kk+": "+"empty" );
			}
		}
		System.out.println("Total Word count: "+ wordCount );
	 }
	 
	 public void printTopWord2(int numWords){
		 
			//trimTopics();
			
			//int[] ak = (int[]) ArrayUtils.asPrimitiveArray(nk);
			//Arrays.sort(ak);
			
			int wordCount=0;
			
			for(int k=0 ; k<(K-1); k++){
					
				if(nk.get(k)!=0){
					
					//check word count
					wordCount+=nk.get(k);
					
					IDSorter[] sortedTypes = new IDSorter[numTypes];
					//sort word in topic k
					for (int type=0; type < numTypes; type++) {
						sortedTypes[type] = new IDSorter(type, nkt.get(k)[type]);
					}
					
					Arrays.sort(sortedTypes);
				
					Alphabet alphabet = instances.getDataAlphabet();
					StringBuffer out = new StringBuffer();
					out.append("topic"+k + ": ");
					out.append("word:"+ nk.get(k) + ", ");
					double prop = (double)nk.get(k)/totalWord;
					out.append(String.format("prop:%2.4f, ", prop));
				
					for (int i=0; i<numWords; i++) {
						out.append(alphabet.lookupObject(sortedTypes[i].getID()) + " ");
					}
					System.out.println(out);
				}
				else{
					System.out.println("Topic"+k+": "+"empty" );
				}
			}
			System.out.println("Total Word count: "+ wordCount );
		 }
	 
	 public void printParameter(){
		 
		 String out = String.format("Summary: Docs=%d, topics=%d, totalWords=%d, alpha=%2.3f, beta=%2.3f, gamma=%2.3f",numDocuments, (K-1),totalWord, alpha, beta, gamma);
		 System.out.println(out);
	 
	 }
	 
	 public void trimTopics(){
		 
		 System.out.println("start trim");
		 
		 int[] new_nk = IndexQuickSort.sort(nk);
		 IndexQuickSort.reverse(new_nk);
		 
		 //remove empty topic
		 IndexQuickSort.reorder(nk, new_nk);
		 IndexQuickSort.reorder(nkt, new_nk);
		 
		 for(int i=0; i < kgaps.size(); i++){
			 nk.remove(nk.size()-1 );
			 nkt.remove(nkt.size()-1 );
		 }
		 
		 for(int m=0 ; m < numDocuments; m++){
			 IndexQuickSort.reorder(nmk[m], new_nk);
			 //remove gaps
			 for(int i=0 ; i <kgaps.size() ; i++){
				 nmk[m].remove(nmk[m].size()-1);
			 }
		 }
		 //clean kgaps
		 kgaps.clear();
		 int[] k2knew = IndexQuickSort.inverse(new_nk);
		 
		 //rewrite topic
		 for(int i=0; i<K; i++){
			 kactive.set(i, k2knew[kactive.get(i)]);
		 }
		 
		 for(int m=0; m<numDocuments ; m++){
			 FeatureSequence fs = (FeatureSequence) instances.get(m).getData();
			 
			 for(int n=0 ; n<fs.getLength() ; n++){
				 z[m][n]=k2knew[z[m][n]]; 
			 }
		 }
		 
		 System.out.println("end trim");
	 }
	 
	 /*preplexity*/
	 public double getPreplexity(){
		 
		 double preplexity=0.0;
		 
		 return preplexity;
	 }
}
