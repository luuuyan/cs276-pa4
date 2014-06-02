package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMOreg;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class SVRLearner extends Learner {

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		
		/*
		 * @TODO: Below is a piece of sample code to show 
		 * you the basic approach to construct a Instances 
		 * object, replace with your implementation. 
		 */
		Map<Query, List<Document>> trainData;
		Map<String, Map<String, Double>> relData;
		try {
			trainData = Util.loadTrainData(train_data_file);
			relData = Util.loadRelData(train_rel_file);
			
			
			Instances dataset = null;
			
			/* Build attributes list */
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			attributes.add(new Attribute("url_w"));
			attributes.add(new Attribute("title_w"));
			attributes.add(new Attribute("body_w"));
			attributes.add(new Attribute("header_w"));
			attributes.add(new Attribute("anchor_w"));
			attributes.add(new Attribute("relevance_score"));
			dataset = new Instances("train_dataset", attributes, 0);
			
			/* Add data */
			
			// TO Do: add data to the instances
			for (Query q : trainData.keySet()){
				for (Document d : trainData.get(q)){
					double[] values = new double[6];
					double[] tdidfs = AScorer.getTfIdf(d, q, idfs);
					System.arraycopy(tdidfs, 0, values, 0, tdidfs.length);
					values[5] = relData.get(q.toString().toLowerCase()).get(d.url);
					// add data
					Instance inst = new DenseInstance(1.0, values);
					dataset.add(inst);
				}
			}
			
			/* Set last attribute as target */
			dataset.setClassIndex(dataset.numAttributes() - 1);
			return dataset;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}

	@Override
	public Classifier training(Instances dataset) {
		/*
		 * @TODO: Your code here
		 */
		SMOreg model = new SMOreg();
		try {
			model.buildClassifier(dataset);
			return model;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		TestFeatures featureDataSet = new TestFeatures();
		Map<String, Map<String, Integer>> indexMap = new HashMap<String, Map<String, Integer>>();
		int totalCount = 0;

		try {
			Map<Query, List<Document>> testData;
			testData = Util.loadTrainData(test_data_file);			
			
			Instances dataset = null;
			
			/* Build attributes list */
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			attributes.add(new Attribute("url_w"));
			attributes.add(new Attribute("title_w"));
			attributes.add(new Attribute("body_w"));
			attributes.add(new Attribute("header_w"));
			attributes.add(new Attribute("anchor_w"));
			attributes.add(new Attribute("relevance_score"));
			dataset = new Instances("test_dataset", attributes, 0);
			
			/* Add data */
			
			// TO Do: add data to the instances
			for (Query q : testData.keySet()){
				if (!indexMap.containsKey(q.query)){
					indexMap.put(q.query, new HashMap<String, Integer>());
				}
				for (Document d : testData.get(q)){
					double[] values = new double[6];
					double[] tdidfs = AScorer.getTfIdf(d, q, idfs);
					System.arraycopy(tdidfs, 0, values, 0, tdidfs.length);
					values[5] = -1.0; // relevance score not used in prediction
					
					// add data to dataset
					Instance inst = new DenseInstance(1.0, values);
					dataset.add(inst);
					
					// add index to indexmap
					indexMap.get(q.query).put(d.url, totalCount);
					totalCount ++;
				}
			}
			
			/* Set last attribute as target */
			dataset.setClassIndex(dataset.numAttributes() - 1);
			
			featureDataSet.features = dataset;
			featureDataSet.index_map = indexMap;
			
			return featureDataSet;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
		 */
		Map<String, List<String>> rankedResult = new HashMap<String, List<String>>();
		Instances test_dataset = tf.features;
		Map<String, Map<String, Integer>> indexMap = tf.index_map;
		try {
			for (String queryString : indexMap.keySet()){
				ArrayList<Pair<String, Double>> urlAndScoreList = new ArrayList<Pair<String, Double>>();
				for (String url : indexMap.get(queryString).keySet()){
					int index = indexMap.get(queryString).get(url);
					double prediction = model.classifyInstance(test_dataset.instance(index));
					Pair<String, Double> urlAndScore = new Pair<String, Double>(url, prediction);
					urlAndScoreList.add(urlAndScore);
				}
				
				// sort the urls based on the predicted relevant score
				Collections.sort(urlAndScoreList, new Comparator<Pair<String, Double>>() {
					public int compare(Pair<String, Double> p1, Pair<String, Double> p2){
						if(p1.getSecond() < p2.getSecond()){
							return 1;
						}else{
							return -1;
						}
					}
				});
				
				List<String> rankedUrls = new ArrayList<String>();
				for (Pair<String, Double> pair : urlAndScoreList){
					rankedUrls.add(pair.getFirst());
				}
				
				rankedResult.put(queryString, rankedUrls);
			}
			return rankedResult;

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}

}
