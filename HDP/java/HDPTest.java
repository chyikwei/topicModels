import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.regex.Pattern;

import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.CharSequenceLowercase;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.topics.HDP;
import cc.mallet.topics.HDPInferencer;
import cc.mallet.types.InstanceList;


public class HDPTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		
		//input file, one file for training, one for test
		String inputFileName = "data/ISI_Abstract_train.txt";
		String testFileName = "data/ISI_Abstract_test.txt";				
				
		// Begin by importing documents from text to feature sequences
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();

		// Pipes: lowercase, tokenize, remove stopwords, map to features
		pipeList.add( new CharSequenceLowercase() );
		//word format by Regular expression
		pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")) );
		pipeList.add( new TokenSequenceRemoveStopwords(new File("stoplist/en.txt"), "UTF-8", false, false, false) );
		//add bigram words
		//pipeList.add(new TokenSequenceNGrams(new int[] {2} ));
				
		//convert to feature
		pipeList.add( new TokenSequence2FeatureSequence() );

		InstanceList instances = new InstanceList (new SerialPipes(pipeList));
		InstanceList testInstances = new InstanceList (instances.getPipe());
				
		Reader insfileReader = new InputStreamReader(new FileInputStream(new File(inputFileName)), "UTF-8");
		Reader testfileReader = new InputStreamReader(new FileInputStream(new File(testFileName)), "UTF-8");
				
		instances.addThruPipe(new CsvIterator (insfileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
													   3, 2, 1)); // data, label, name fields
		testInstances.addThruPipe(new CsvIterator (testfileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
				   3, 2, 1)); // data, label, name fields
		
		//setup HDP parameters(alpha, beta, gamma, initialTopics)
		HDP hdp = new HDP(1.0, 0.1, 1.0, 10);
		hdp.initialize(instances);
		
		//set number of iterations, and display result or not 
		hdp.estimate(2000);
		
		//get topic distribution for first instance
		double[] distr = hdp.topicDistribution(0);
		//print out
		for(int j=0; j<distr.length ; j++){
			System.out.print(distr[j] + " ");
		}
		
		//for inferencer
		HDPInferencer inferencer = hdp.getInferencer();
		inferencer.setInstance(testInstances);
		inferencer.estimate(100);
		//get topic distribution for first test instance
		distr = inferencer.topicDistribution(0);		
		//print out
		for(int j=0; j<distr.length ; j++){
			System.out.print(distr[j] + " ");
		}
		//get preplexity
		double prep = inferencer.getPreplexity();
		System.out.println("preplexity for the test set=" + prep);
		
		//10-folds cross validation, with 1000 iteration for each test.
		hdp.runCrossValidation(10, 1000);
		
	}

}