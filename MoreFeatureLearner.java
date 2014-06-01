package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class MoreFeatureLearner extends Learner {
  private LibSVM model;
  private BM25Scorer bm25scorer;
  private SmallestWindowScorer windowScorer;
  
  public MoreFeatureLearner(Map<String, Double> idf){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
  }
  
  	final public String separator = "###";
  	
  	private static void addAttributes(List<Attribute> attributes) {
  		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("bm25"));
		attributes.add(new Attribute("window"));
		attributes.add(new Attribute("pagerank"));
  	}
  	
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		Map<Query, List<Document>> originData;
		Map<String, Map<String, Double>> relData;
		try {
			originData = Util.loadTrainData(train_data_file);
			relData = Util.loadRelData(train_rel_file);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		
		bm25scorer = new BM25Scorer(idfs, originData);
		windowScorer = new SmallestWindowScorer(idfs, originData);
			
		// first build originInstances
		ArrayList<Attribute> origin_attributes = new ArrayList<Attribute>();
		addAttributes(origin_attributes);
		origin_attributes.add(new Attribute("rel_score"));
		Instances originInstances = new Instances("origin_dataset", origin_attributes, 0);

		// originIndexMap query -> url -> index in originInstances/normalizeInstances
		Map<String, Map<String, Integer>> originIndexMap = new HashMap<String, Map<String, Integer>>();
		int countInOriginInstances = 0;
		int numAttributes = originInstances.numAttributes();
		
		// add data
		for (Query q : originData.keySet()){
			originIndexMap.put(q.query, new HashMap<String, Integer>());
			for (Document d : originData.get(q)){
				double[] values = new double[numAttributes];
				double[] tdidfs = AScorer.getTfIdf(d, q, idfs);
				System.arraycopy(tdidfs, 0, values, 0, tdidfs.length);
				values[5] = bm25scorer.getSimScore(d, q);
				values[6] = windowScorer.getSimScore(d, q);
				values[7] = d.page_rank;
				values[numAttributes-1] = relData.get(q.toString().toLowerCase()).get(d.url);
				// add data
				Instance inst = new DenseInstance(1.0, values);
				originInstances.add(inst);
				// record in originIndexMap
				originIndexMap.get(q.query).put(d.url, countInOriginInstances);
				countInOriginInstances ++;
			}
		}
		
		/* Set last attribute as target */
		originInstances.setClassIndex(originInstances.numAttributes() - 1);
		
		// normalize originInstances and get normalizeInstances
		Standardize filter = new Standardize();
		Instances normalizeInstances = null;
		try {
			filter.setInputFormat(originInstances);
			normalizeInstances = Filter.useFilter(originInstances, filter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		normalizeInstances.setClassIndex(originInstances.numAttributes() - 1);

		
		// use normalizeInstances to derive trainInstances (with different of tfidf as features)
		Instances trainInstances = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		addAttributes(attributes);
		// label: +1 or -1
		ArrayList<String> labels = new ArrayList<String>();
		labels.add("+1");
		labels.add("-1");
		attributes.add(new Attribute("diff_rel_score", labels));
		trainInstances = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		for (Query q : originData.keySet()){
			int docSize = originData.get(q).size();
			for (int i = 0; i < docSize - 1; i++){
				for (int j = i + 1; j < docSize; j++){
					double[] values1 = new double[numAttributes];
					double[] values2 = new double[numAttributes];
					String d1 = originData.get(q).get(i).url;
					String d2 = originData.get(q).get(j).url;
					Instance i1 = normalizeInstances.instance(originIndexMap.get(q.query).get(d1));
					Instance i2 = normalizeInstances.instance(originIndexMap.get(q.query).get(d2));

					// features xi - xj
					for (int k = 0; k < numAttributes; k++){
						values1[k] = i1.value(k) - i2.value(k);
						values2[k] = i2.value(k) - i1.value(k);
					}

					// diff rel score +1 or -1
					int predidx = numAttributes - 1;
					if (values1[predidx] > 0){
						values1[predidx] = trainInstances.attribute(predidx).indexOfValue("+1");
						values2[predidx] = trainInstances.attribute(predidx).indexOfValue("-1");
					}else {
						values1[predidx] = trainInstances.attribute(predidx).indexOfValue("-1");
						values2[predidx] = trainInstances.attribute(predidx).indexOfValue("+1");
					}
					// add data
					Instance inst1 = new DenseInstance(1.0, values1);
					Instance inst2 = new DenseInstance(1.0, values2);
					trainInstances.add(inst1);
					trainInstances.add(inst2);
				}
			}	
		}
		
		/* Set last attribute as target */
		trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
		
		
//			System.out.println("before normalized instances: " + dataset.toString());
//			System.out.println("normalized instances: " + standardize_X.toString());
		return trainInstances;

//			return dataset;
		
	}

	@Override
	public Classifier training(Instances dataset) {
		/*
		 * @TODO: Your code here
		 */

		try {
			model.buildClassifier(dataset);
			return model;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		TestFeatures featureDataSet = new TestFeatures();
		// query -> "d1;d2" -> index
		Map<String, Map<String, Integer>> indexMap = new HashMap<String, Map<String, Integer>>();
		int totalCount = 0;

		try {
			Map<Query, List<Document>> originData = Util.loadTrainData(test_data_file);	// origin testdata
			bm25scorer = new BM25Scorer(idfs, originData);
			windowScorer = new SmallestWindowScorer(idfs, originData);
			
			// first build originInstances
			ArrayList<Attribute> origin_attributes = new ArrayList<Attribute>();
			addAttributes(origin_attributes);
			origin_attributes.add(new Attribute("rel_score"));
			Instances originInstances = new Instances("origin_dataset", origin_attributes, 0);

			// originIndexMap query -> url -> index in originInstances/normalizeInstances
			Map<String, Map<String, Integer>> originIndexMap = new HashMap<String, Map<String, Integer>>();
			int countInOriginInstances = 0;
			int numAttributes = originInstances.numAttributes();
			
			// add data
			for (Query q : originData.keySet()){
				originIndexMap.put(q.query, new HashMap<String, Integer>());
				for (Document d : originData.get(q)){
					double[] values = new double[numAttributes];
					double[] tdidfs = AScorer.getTfIdf(d, q, idfs);
					System.arraycopy(tdidfs, 0, values, 0, tdidfs.length);
					values[5] = bm25scorer.getSimScore(d, q);
					values[6] = windowScorer.getSimScore(d, q);
					values[7] = d.page_rank;
					values[numAttributes-1] = -1; // not relevant for testing
					// add data
					Instance inst = new DenseInstance(1.0, values);
					originInstances.add(inst);
					// record in originIndexMap
					originIndexMap.get(q.query).put(d.url, countInOriginInstances);
					countInOriginInstances ++;
				}
			}
			
			/* Set last attribute as target */
			originInstances.setClassIndex(originInstances.numAttributes() - 1);
			
			// normalize originInstances and get normalizeInstances
			Standardize filter = new Standardize();
			filter.setInputFormat(originInstances);
			Instances normalizeInstances = Filter.useFilter(originInstances, filter);
			normalizeInstances.setClassIndex(originInstances.numAttributes() - 1);
		
			// derive testInstances from normalizeInstances
			Instances testInstances = null;
			
			/* Build attributes list */
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			addAttributes(attributes);
			// label: +1 or -1
			ArrayList<String> labels = new ArrayList<String>();
			labels.add("+1");
			labels.add("-1");
			attributes.add(new Attribute("diff_rel_score", labels));
			testInstances = new Instances("test_dataset", attributes, 0);
			
			/* Add data */
			
			// TO Do: add data to the instances
			for (Query q : originData.keySet()){
				if (!indexMap.containsKey(q.query)){
					indexMap.put(q.query, new HashMap<String, Integer>());
				}
				for (int i = 0; i < originData.get(q).size() - 1; i++){
					for (int j = i + 1; j < originData.get(q).size(); j++){
						double[] values1 = new double[numAttributes];
						String d1 = originData.get(q).get(i).url;
						String d2 = originData.get(q).get(j).url;
						Instance i1 = normalizeInstances.instance(originIndexMap.get(q.query).get(d1));
						Instance i2 = normalizeInstances.instance(originIndexMap.get(q.query).get(d2));

						// features xi - xj
						for (int k = 0; k < numAttributes; k++){
							values1[k] = i1.value(k) - i2.value(k);
//							values2[k] = i2.value(k) - i1.value(k);
						}

						values1[numAttributes-1] = testInstances.attribute(numAttributes-1).indexOfValue("+1"); // non-relevant here
						Instance inst1 = new DenseInstance(1.0, values1);
						testInstances.add(inst1);
						
						// add index to indexmap
						indexMap.get(q.query).put(d1 + separator + d2, totalCount);
						totalCount ++;
					}
				}	
				
			}
			
//			System.out.println("+1 index : " + testInstances.attribute(5).indexOfValue("+1") + "-1 index : " + testInstances.attribute(5).indexOfValue("-1"));
			
			/* Set last attribute as target */
			testInstances.setClassIndex(testInstances.numAttributes() - 1);
			
			featureDataSet.features = testInstances;
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
		Map<String, String> relScoreMap = new HashMap<String, String>();  // relScoreMap: "url1###url2" -> (predicted) label
		try {
			for (String queryString : indexMap.keySet()){
				// add all the urls w.r.t to this query into a list
				Set<String> docSet = new HashSet<String>();
				for (String urls : indexMap.get(queryString).keySet()){
					// a pair of urls
					String[] urlPair = urls.split(separator);
					if (urlPair.length != 2){
						throw new Exception("Every instance represent the diff of features of a pair url. Now the number of urls for this instance is : " + urlPair.length + " and the urls are : " + urls);
					}
					docSet.add(urlPair[0]);
					docSet.add(urlPair[1]);
					
					// add prediction score of this pairs to relScoreMap
//					System.out.println("classify result:" + model.classifyInstance(test_dataset.instance(indexMap.get(queryString).get(urls))));
//					System.out.println("class 1.0 : " + test_dataset.attribute(5).value(1) + "class 0.0 : " + test_dataset.attribute(5).value(0));
					int result = (int) model.classifyInstance(test_dataset.instance(indexMap.get(queryString).get(urls)));
					String label = test_dataset.attribute(test_dataset.numAttributes()-1).value(result);
					relScoreMap.put(urls, label);
				}
				
				List<SortableUrl> docList = new ArrayList<SortableUrl>();
				for (String url : docSet){
					docList.add(new SortableUrl(relScoreMap, url));
				}
				
				// sort the list of urls according to the SVM rel score;
				Collections.sort(docList);
				List<String> rankedUrls = new ArrayList<String>();
				
				for (SortableUrl sortableUrl : docList){
					rankedUrls.add(sortableUrl.url);
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

	public class SortableUrl implements Comparable<SortableUrl>{
		// "url1###url2" -> label (+1 or -1)
		Map<String, String> relScoreMap;
		String url;
		public SortableUrl(Map<String, String> relScoreMap, String url){
			this.relScoreMap = relScoreMap;
			this.url = url;
		}

		@Override
		public int compareTo(SortableUrl o2) {
			// TODO Auto-generated method stub
			if (relScoreMap.containsKey(url + separator + o2.url)){
				return (relScoreMap.get(url + separator + o2.url).equals("+1"))? -1 : 1;
			}else if (relScoreMap.containsKey(o2.url + separator + url)){
				return (relScoreMap.get(o2.url + separator + url).equals("-1"))? -1 : 1;
			}else {
				// error, should not fall in this
				System.out.println("error: neither url1:url2 nor url2:url1 in the relScoreMap");
				return 0;
			}
		}
	}
}
