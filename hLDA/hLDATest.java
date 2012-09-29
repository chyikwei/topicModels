import cc.mallet.util.*;
import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.topics.*;

import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.*;
import java.io.*;

public class hLDATest {

	public static void main(String[] args) throws Exception {

		// Begin by importing documents from text to feature sequences
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();

		// Pipes: lowercase, tokenize, remove stopwords, map to features
		pipeList.add( new CharSequenceLowercase() );
		//pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}\\p{N}]+[\\p{L}\\p{N}]")) );
		pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}-_\\p{N}]+[\\p{L}\\p{N}]")) );
		pipeList.add( new TokenSequenceRemoveStopwords(new File("stoplist/en.txt"), "UTF-8", false, false, false) );
		pipeList.add( new TokenSequence2FeatureSequence() );

		InstanceList instances = new InstanceList (new SerialPipes(pipeList));
		InstanceList testInstances = new InstanceList (instances.getPipe());
		
		String inputFileName = "data/ISI_Abstract_original.txt";
		Reader insfileReader = new InputStreamReader(new FileInputStream(new File(inputFileName)), "UTF-8");
		//Reader testfileReader = new InputStreamReader(new FileInputStream(new File(args[1])), "UTF-8");
		
		instances.addThruPipe(new CsvIterator (insfileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
											   3, 2, 1)); // data, label, name fields
		//testInstances.addThruPipe(new CsvIterator (testfileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
		//		   3, 2, 1)); // data, label, name fields

		HierarchicalLDA model = new HierarchicalLDA();
        
		//set parameter
		model.setAlpha(10.0);
		model.setGamma(1.0);
		model.setEta(0.1);
		
		//set level
		model.initialize(instances, instances, 3, new Randoms());

		model.estimate(3000);
		
		HierarchicalLDAInferencer inferencer = new HierarchicalLDAInferencer(model);
		
		inferencer.printNode(inferencer.rootNode, 0);
		
		try{
	       // print tree structure 
		   //FileWriter fstream = new FileWriter("raw-hlda-3level-topicTree_eta07.csv");
		   FileWriter fstream = new FileWriter("output.csv");
		   BufferedWriter out = new BufferedWriter(fstream);
		   inferencer.printNodeTofile(inferencer.rootNode, 0, out);
				
		   //Close the output stream
		   out.close();
		
		   System.out.println("total nodes:" + inferencer.counter);
		   System.out.println();
		   
		   // print train instance 
		   //BufferedWriter trainOut = new BufferedWriter(new FileWriter("raw-hlda-3level-topic_distribution_eta07.csv"));
		   BufferedWriter trainOut = new BufferedWriter(new FileWriter("outpuut2")); 
		   inferencer.printTrainData(instances, trainOut);
		   trainOut.close();
		   
		   System.out.println("total train instance:" + instances.size());
		   System.out.println();
		   
		   
		   // print test instance   
		   //BufferedWriter testOut = new BufferedWriter(new FileWriter("hlda-test-v5-4level-5000.csv"));
		   //inferencer.printTestData(testInstances, 300, testOut);
		   //testOut.close();
		   
		   System.out.println("total test instance:" + testInstances.size());
		   	  
		   
		}catch(Exception e){
			e.printStackTrace();
		}
		
	}

}