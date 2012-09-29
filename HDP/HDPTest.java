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
import cc.mallet.types.InstanceList;


public class HDPTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		//input file
		String inputFileName = "data/ISI_Abstract_original.txt";
						
				
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
		//Reader testfileReader = new InputStreamReader(new FileInputStream(new File(args[1])), "UTF-8");
				
		instances.addThruPipe(new CsvIterator (insfileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
													   3, 2, 1)); // data, label, name fields
		//testInstances.addThruPipe(new CsvIterator (testfileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
		//		   3, 2, 1)); // data, label, name fields
		
		HDP hdp = new HDP(1.0, 0.4, 1.0, 5);
		hdp.initialize(instances);
		hdp.estimate(500, true);
		
		hdp.trimTopics();
		
		System.out.println();
		
		hdp.printTopWord2(6);

	}

}
